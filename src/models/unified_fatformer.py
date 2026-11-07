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


def router_is_approved(params, proof):
    digest = hashlib.sha256(json.dumps(params, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return bool(proof and proof.get("schema") == "router_ablation/v2"
                and proof.get("evidence_level") == "measured"
                and proof.get("params_digest") == digest
                and proof.get("checkpoint_sha256") == params["source"]["checkpoint_sha256"]
                and proof.get("bootstrap", {}).get("n", 0) >= 1000
                and proof.get("verdict", {}).get("criteria_met") is True
                and proof["verdict"].get("wins", 0) >= 3
                and proof["verdict"].get("ci_separated", 0) >= 2)


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
