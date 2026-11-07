"""Single-backbone, single-forward inference with measured calibration and safe routing.

Clean/forensic share logits and differ only by condition-calibrated temperature.
This is not a second forensic model or a claim that the two experts are independent.
"""
import hashlib
import json
import math
import warnings

import torch
from torch import nn

from .degradation_router import DegradationRouter, forward_with_gating
from .gating import DynamicFrequencyGating

CFE_SUBSETS = ("1_commercial", "2_latent_diffusion", "3_pixel_diffusion",
               "4_modern_gans", "5_other_autoregressive")


def is_sha256_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value, lower, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError("Invalid finite report metric")
    return value


def _count(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid report count")
    return value


def _interval(value, lower, upper):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Invalid confidence interval")
    lo, hi = (_number(v, lower, upper) for v in value)
    if lo > hi:
        raise ValueError("Reversed confidence interval")
    return lo, hi


def validate_router_report(params, proof):
    """Validate a complete report and recompute its verdict from all five subsets.

    Hashes bind referenced artifacts and parameters; they are not digital signatures
    or independent verification that the underlying measurements are genuine.
    Runtime does not reopen evaluation data or rerun bootstrap.
    """
    if not isinstance(params, dict) or not isinstance(proof, dict):
        raise ValueError("Router parameters and report must be objects")
    if (params.get("schema") != "degradation_router/v2" or params.get("status") != "fitted"
            or params.get("evidence_level") != "measured"
            or proof.get("schema") != "router_ablation/v2" or proof.get("status") != "evaluated"
            or proof.get("evidence_level") != "measured"):
        raise ValueError("Require fitted parameters and an evaluated measured report")
    digest = hashlib.sha256(json.dumps(params, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if proof.get("params_digest") != digest:
        raise ValueError("Report does not match router parameters")
    source = params["source"]
    for report_key, expected in (("checkpoint_sha256", source["checkpoint_sha256"]),
                                 ("calibration_cache_sha256", source["cache_sha256"])):
        if not is_sha256_digest(expected) or proof.get(report_key) != expected:
            raise ValueError("Report checkpoint/calibration fingerprint mismatch")
    for key in ("params_file_sha256", "evaluation_cache_sha256", "evaluation_split_sha256"):
        if not is_sha256_digest(proof.get(key)):
            raise ValueError("Missing artifact fingerprint")
    if (proof.get("preprocess") != params["preprocess"]
            or proof.get("branch_design") != params["branch_design"]):
        raise ValueError("Report preprocessing/branch design mismatch")
    bootstrap = proof["bootstrap"]
    _count(bootstrap["n"], 1000)
    _count(bootstrap["seed"])
    if (bootstrap.get("paired") is not True
            or bootstrap.get("unit") != "source image, both views retained"
            or bootstrap.get("strata") != "real_fake_source_images"
            or _number(bootstrap.get("confidence_level"), 0, 1) != .95):
        raise ValueError("Wrong bootstrap protocol")
    subsets = proof["subsets"]
    if not isinstance(subsets, dict) or set(subsets) != set(CFE_SUBSETS):
        raise ValueError("Report must contain exactly five CFE subsets")
    wins = separated = 0
    for row in subsets.values():
        n = _count(row["n_source_images"], 2)
        if (_count(row["n_views"], 4) != 2*n
                or _count(row["n_real_source_images"], 1) + _count(row["n_fake_source_images"], 1) != n):
            raise ValueError("Inconsistent paired-view sample counts")
        for name in ("fixed", "router", "oracle"):
            result = row[name]
            _number(result["fake_at_real90"], 0, 100)
            _interval(result["ci95"], 0, 100)
            if set(result["by_condition"]) != {"clean", "q30"}:
                raise ValueError("Missing condition metrics")
            for value in result["by_condition"].values():
                _number(value, 0, 100)
        delta = row["router"]["fake_at_real90"] - row["fixed"]["fake_at_real90"]
        if not math.isclose(_number(row["delta_pp"], -100, 100), delta, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Declared delta differs from measured scores")
        _interval(row["delta_ci95"], -100, 100)
        win = delta >= 1.
        above = row["router"]["ci95"][0] > row["fixed"]["ci95"][1]
        if row["win"] is not win or row["router_ci_above_fixed"] is not above:
            raise ValueError("Subset flags differ from scores/confidence intervals")
        wins += win
        separated += above
    won = wins >= 3 and separated >= 2
    verdict = proof["verdict"]
    if (any(_count(verdict[key]) != value for key, value in
            (("wins", wins), ("ci_separated", separated), ("total_subsets", 5)))
            or verdict["criteria_met"] is not won or verdict["fallback_required"] is not (not won)):
        raise ValueError("Declared verdict differs from recomputed five-subset verdict")
    return dict(wins=wins, total_subsets=5, ci_separated=separated,
                criteria_met=won, fallback_required=not won)


def router_is_approved(params, proof):
    try:
        return validate_router_report(params, proof)["criteria_met"]
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        # A partial/malformed report never enables routing; calibrated fixed fallback remains available.
        return False


class UnifiedFatFormerXLA(nn.Module):
    def __init__(self, backbone, calibration=None, router_params=None, router_ablation=None,
                 checkpoint_sha256=None):
        super().__init__()
        self.backbone = backbone
        self.router = None
        self.router_enabled = False
        self.register_buffer("clip_mean", torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1))
        self.register_buffer("clip_std", torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1))
        if not checkpoint_sha256:
            raise ValueError("Pass SHA-256 of the canonical-loaded backbone checkpoint")
        self.checkpoint_sha256 = checkpoint_sha256
        if (router_params and router_params.get("status") == "fitted"
                and router_params.get("schema") == "degradation_router/v2"):
            if router_params["source"]["checkpoint_sha256"] != checkpoint_sha256:
                raise ValueError("Router calibration belongs to another checkpoint")
            self.router = DegradationRouter.from_params(router_params)
            self.clean_temperature = float(router_params["calibration"]["clean"]["T"])
            self.forensic_temperature = float(router_params["calibration"]["forensic"]["T"])
            self.router_enabled = router_is_approved(router_params, router_ablation)
            self.router.fallback_mode = not self.router_enabled
            self.fixed_branch = router_params["fixed"]["branch"]
            self.fixed_threshold = router_params["fixed"]["threshold"]
        else:
            if (not calibration or calibration.get("schema") != "calibration/v1"
                    or calibration.get("evidence_level") != "measured"
                    or calibration.get("checkpoint", {}).get("sha256") != checkpoint_sha256):
                raise ValueError("Missing valid measured fallback calibration for this checkpoint")
            self.clean_temperature = self.forensic_temperature = float(calibration["T"])
            threshold = calibration.get("T_star_default", calibration.get("T_star"))
            if isinstance(threshold, dict):
                threshold = threshold["0.05"]
            self.fixed_threshold = float(threshold)
            self.fixed_branch = "clean"
        if (not all(math.isfinite(v) and v > 0 for v in (self.clean_temperature, self.forensic_temperature))
                or not 0 <= self.fixed_threshold <= 1.000001):
            raise ValueError("Invalid calibrated temperatures/threshold")
        if not self.router_enabled:
            warnings.warn("[WARN] router disabled: no validated held-out win; calibrated fixed fallback", RuntimeWarning)

    def get_extra_state(self):
        return {key: getattr(self, key) for key in ("checkpoint_sha256", "router_enabled", "clean_temperature",
                                                   "forensic_temperature", "fixed_branch", "fixed_threshold")}

    def set_extra_state(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    @torch.no_grad()
    def forward(self, x):
        if self.training or self.backbone.training:
            raise ValueError("Unified inference requires eval()")
        lam = None
        if self.router_enabled and any(isinstance(m, DynamicFrequencyGating) for m in self.backbone.modules()):
            logits, lam = forward_with_gating(self.backbone, x)
        else:
            logits = self.backbone(x)
            if isinstance(logits, tuple):
                logits = logits[0]
        pc = torch.softmax(logits.float()/self.clean_temperature, -1)[:, 1]
        pf = torch.softmax(logits.float()/self.forensic_temperature, -1)[:, 1]
        if self.router is not None:
            raw = (x*self.clip_std+self.clip_mean).clamp(0,1)
            p, label, w, threshold = self.router(raw, pc, pf, lam)
        else:
            p = pc if self.fixed_branch == "clean" else pf
            threshold = torch.full_like(p, self.fixed_threshold)
            label = (p >= threshold).long()
            w = torch.full_like(p, float(self.fixed_branch == "forensic"))
        return dict(prob_fake=p, label=label, w_deg=w, threshold=threshold)

    predict = forward
