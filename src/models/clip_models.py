import torch
import torch.nn as nn
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)[1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class LanguageGuidedAlignment(nn.Module):
    def __init__(self, clip_model, classnames=["real", "synthetic"], args=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = getattr(args, "num_context_embedding", 8) if args else 8
        ctx_init = getattr(args, "init_context_embedding", "") if args else ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        # BẮT BUỘC giữ nguyên lỗi chính tả patch_basaed_enhancer để tương thích 100% với checkpoint gốc
        d_model = clip_model.ln_final.weight.shape[0]
        d_ffn = d_model * 4
        self.patch_basaed_enhancer = nn.MultiheadAttention(d_model, num_heads=12)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        return tgt

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        tgt = ctx[:, None].repeat_interleave(im_features.shape[0], dim=1)
        tgt2 = self.patch_basaed_enhancer(tgt, im_features.transpose(0, 1), im_features.transpose(0, 1))[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt = self.forward_FFN(tgt)

        ctx_shifted = tgt.transpose(0, 1)

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        return prompts


class CLIPModel(nn.Module):
    def __init__(self, name="ViT-L/14", args=None):
        super(CLIPModel, self).__init__()
        self.clip_model = clip.load(name, device="cpu", args=args)[0]
        self.language_guided_alignment = LanguageGuidedAlignment(self.clip_model, classnames=["real", "fake"], args=args)

        self.tokenized_prompts = self.language_guided_alignment.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.num_classes = getattr(args, "num_classes", 2) if args else 2

        d_model = self.clip_model.ln_final.weight.shape[0]
        d_ffn = d_model * 4
        self.text_guided_interactor = nn.MultiheadAttention(d_model, num_heads=12)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_FFN(self, tgt):
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype), return_full=True)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.language_guided_alignment(image_features)

        # Eq. (1): Vanilla CLIP similarity S(i)
        logits = []
        text_feature_list = []
        for pts_i, imf_i in zip(prompts, image_features_norm):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_feature_list.append(text_features)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i[0] @ text_features_norm.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        # Eq. (9): Text-guided interactor similarity S'(i)
        text_features = torch.stack(text_feature_list, dim=1)
        tgt = image_features[:, 1:].transpose(0, 1)
        tgt2 = self.text_guided_interactor(tgt, text_features, text_features)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt = self.forward_FFN(tgt)

        aug_image_features = tgt.transpose(0, 1).mean(dim=1)
        aug_image_features = aug_image_features / aug_image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        aug_logits = []
        for pts_i, imf_i in zip(aug_image_features, text_features_norm.transpose(0, 1)):
            aug_logits.append(logit_scale * pts_i @ imf_i.t())
        aug_logits = torch.stack(aug_logits)

        # Trả về logits tổng hợp S(i) + S'(i)
        return logits + aug_logits
