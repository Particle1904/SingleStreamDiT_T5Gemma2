import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import Config

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash-Attn Available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash-Attn NOT Available")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normed * self.weight.to(x.dtype)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size), 
            nn.SiLU(), 
            nn.Linear(hidden_size, hidden_size)
        )
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, 128).float() / 128)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * 1000.0
        freqs = self.inv_freq.to(device=t.device)
        args = t_scaled[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))

class SwiGLUWithImageConv(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True)

    def forward(self, x: torch.Tensor, seq_len_text: int, grid_h: int, grid_w: int, run_conv: bool = True) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        gated = F.silu(x1) * x2

        if run_conv and grid_h > 0 and grid_w > 0:
            if seq_len_text == 0:
                B, L, C = gated.shape
                img_spatial = gated.transpose(1, 2).reshape(B, C, grid_h, grid_w)
                img_spatial = self.dwconv(img_spatial)
                gated = img_spatial.reshape(B, C, L).transpose(1, 2)
            else:
                text_part = gated[:, :seq_len_text]
                img_part = gated[:, seq_len_text:]
                B, L, C = img_part.shape
                img_spatial = img_part.transpose(1, 2).reshape(B, C, grid_h, grid_w)
                img_spatial = self.dwconv(img_spatial)
                img_mixed = img_spatial.view(B, C, L).transpose(1, 2)
                gated = torch.cat([text_part, img_mixed], dim=1)

        return self.w3(gated)

class RopeEmbedder(nn.Module):
    def __init__(
        self,
        theta: float = 10000.0,
        axes_dims: List[int] = [32, 16, 16],     
        axes_lens: List[int] = [512, 2048, 2048] 
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        
        cos_list, sin_list = self.precompute_freqs(self.axes_dims, self.axes_lens, theta=self.theta)
        self._freqs_cos = cos_list
        self._freqs_sin = sin_list

    @staticmethod
    def precompute_freqs(dims: List[int], ends: List[int], theta: float):
        cos_list, sin_list = [], []
        for d, e in zip(dims, ends):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            timestep = torch.arange(e, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            cos_list.append(torch.cos(freqs))
            sin_list.append(torch.sin(freqs))
        return cos_list, sin_list

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, num_axes = ids.shape
        assert num_axes == len(self.axes_dims)
        device = ids.device
        
        if self._freqs_cos[0].device != device:
            self._freqs_cos = [f.to(device) for f in self._freqs_cos]
            self._freqs_sin = [f.to(device) for f in self._freqs_sin]
        
        res_cos, res_sin = [], []
        for i in range(num_axes):
            axis_ids = ids[..., i].clamp(0, self._freqs_cos[i].shape[0] - 1).reshape(-1)
            res_cos.append(self._freqs_cos[i][axis_ids].view(B, N, -1))
            res_sin.append(self._freqs_sin[i][axis_ids].view(B, N, -1))
            
        return torch.cat(res_cos, dim=-1), torch.cat(res_sin, dim=-1)

def apply_rotary_emb(x: torch.Tensor, rope_freqs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rope_freqs
    cos = cos.unsqueeze(1) # Shape: (B, 1, N, head_dim // 2)
    sin = sin.unsqueeze(1)
    
    with torch.amp.autocast("cuda", enabled=False):
        x_fp32 = x.float()
        
        x_split = x_fp32.reshape(*x_fp32.shape[:-1], -1, 2)
        x1 = x_split[..., 0]
        x2 = x_split[..., 1]
        
        out_real = x1 * cos - x2 * sin
        out_imag = x1 * sin + x2 * cos
        
        out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
        return out.to(x.dtype)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, n_kv_heads: int, use_adaLN: bool = True, dropout: float = 0.0):
        super().__init__()
        self.use_adaLN = use_adaLN
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // num_heads
        self.num_groups = self.num_heads // self.n_kv_heads

        self.norm1 = RMSNorm(dim)
        self.attn_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.attn_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.attn_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.attn_out = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUWithImageConv(dim, dim * 4)

        if use_adaLN:
            self.adaLN_msa = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
            self.adaLN_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        c: Optional[torch.Tensor] = None, 
        attn_mask: Optional[torch.Tensor] = None, 
        seq_len_text: int = 0, 
        grid_h: int = 0, 
        grid_w: int = 0, 
        rope_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        has_padding: bool = False,
        run_conv: bool = True
    ) -> torch.Tensor:
        B, N, C = x.shape

        if self.use_adaLN and c is not None:
            shift_msa, scale_msa, gate_msa = self.adaLN_msa(c).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_mlp(c).chunk(3, dim=-1)
            
            x_norm = self.norm1(x)
            if seq_len_text > 0:
                x_txt = x_norm[:, :seq_len_text]
                x_img = x_norm[:, seq_len_text:]
                x_img_mod = x_img * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
                x_modulated = torch.cat([x_txt, x_img_mod], dim=1)
            else:
                x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        else:
            x_modulated = self.norm1(x)
            gate_msa = gate_mlp = None

        q = self.attn_q(x_modulated).view(B, N, self.num_heads, self.head_dim)
        k = self.attn_k(x_modulated).view(B, N, self.n_kv_heads, self.head_dim)
        v = self.attn_v(x_modulated).view(B, N, self.n_kv_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if rope_freqs is not None:
            q = apply_rotary_emb(q.transpose(1, 2), rope_freqs).transpose(1, 2)
            k = apply_rotary_emb(k.transpose(1, 2), rope_freqs).transpose(1, 2)

        if FLASH_ATTN_AVAILABLE and q.dtype in (torch.float16, torch.bfloat16) and not has_padding:
            attn_out = flash_attn_func(q, k, v, causal=False, dropout_p=0.0)
        else:
            if self.n_kv_heads != self.num_heads:
                k = k[:, :, :, None, :].expand(B, N, self.n_kv_heads, self.num_groups, self.head_dim).flatten(2, 3)
                v = v[:, :, :, None, :].expand(B, N, self.n_kv_heads, self.num_groups, self.head_dim).flatten(2, 3)
            
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            sdpa_mask = attn_mask.unsqueeze(1).unsqueeze(2) if attn_mask is not None else None
            
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
            attn_out = attn_out.transpose(1, 2)

        attn_out = self.attn_out(attn_out.reshape(B, N, -1))
        
        if gate_msa is not None:
            x = x + gate_msa.unsqueeze(1) * self.dropout(attn_out)
        else:
            x = x + attn_out

        if self.use_adaLN and c is not None:
            x_norm_ffn = self.norm2(x)
            if seq_len_text > 0:
                x_txt_ffn = x_norm_ffn[:, :seq_len_text]
                x_img_ffn = x_norm_ffn[:, seq_len_text:]
                x_img_mod_ffn = x_img_ffn * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
                x_modulated_ffn = torch.cat([x_txt_ffn, x_img_mod_ffn], dim=1)
            else:
                x_modulated_ffn = x_norm_ffn * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        else:
            x_modulated_ffn = self.norm2(x)

        ffn_out = self.ffn(x_modulated_ffn, seq_len_text, grid_h, grid_w, run_conv=run_conv)
        
        if gate_mlp is not None:
            x = x + gate_mlp.unsqueeze(1) * self.dropout(ffn_out)
        else:
            x = x + ffn_out

        return x

class SingleStreamDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = Config.in_channels,
        patch_size: int = Config.patch_size,
        hidden_size: int = Config.hidden_size,
        depth: int = Config.depth,
        num_heads: int = Config.num_heads,
        text_embed_dim: int = Config.text_embed_dim,
        refiner_depth: int = Config.refiner_depth,
        theta: float = Config.rope_base,
        cond_dropout_prob: float = Config.text_dropout,
        max_token_length: int = Config.max_token_length,
        dropout: float = Config.model_dropout
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.n_kv_heads = self.num_heads // 2
        self.text_embed_dim = text_embed_dim
        self.refiner_depth = refiner_depth
        self.rope_theta = theta
        self.cond_dropout_prob = cond_dropout_prob
        self.max_token_length = max_token_length

        patch_sizes = [1, 2]
        self.x_embedders = nn.ModuleDict({
            str(ps): nn.Linear(self.in_channels * ps**2, self.hidden_size) for ps in patch_sizes
        })
        self.final_layers = nn.ModuleDict({
            str(ps): nn.Linear(self.hidden_size, self.in_channels * ps**2) for ps in patch_sizes
        })

        self.cap_embedder = nn.Sequential(
            RMSNorm(self.text_embed_dim), 
            nn.Linear(self.text_embed_dim, self.hidden_size)
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.cap_pad_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        head_dim = self.hidden_size // self.num_heads
        axes_dims = [head_dim // 2, head_dim // 4, head_dim // 4]
        self.rope = RopeEmbedder(theta=self.rope_theta, axes_dims=axes_dims)

        self.rope_cache = {}

        self.noise_refiner = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads, self.n_kv_heads, use_adaLN=True, dropout=dropout) 
            for _ in range(self.refiner_depth)
        ])
        self.context_refiner = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads, self.n_kv_heads, use_adaLN=False, dropout=dropout) 
            for _ in range(self.refiner_depth)
        ])
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads, self.n_kv_heads, use_adaLN=True, dropout=dropout) 
            for _ in range(self.depth)
        ])

        self.final_norm = RMSNorm(self.hidden_size)
        self.initialize_weights()

    def get_rope_freqs(self, seq_len_text: int, grid_h: int, grid_w: int, device: torch.device, p: int, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """O(1) positional coordinate resolution lookup caching."""
        cache_key = (seq_len_text, grid_h, grid_w, device, p)
        if cache_key not in self.rope_cache:
            text_pos = torch.arange(seq_len_text, device=device).unsqueeze(0) 
            zeros = torch.zeros_like(text_pos)
            text_pos_3d = torch.stack([text_pos, zeros, zeros], dim=-1) 
            
            img_start = seq_len_text
            img_len = grid_h * grid_w
            h_coords = torch.arange(grid_h, device=device).repeat_interleave(grid_w) * p
            w_coords = torch.arange(grid_w, device=device).repeat(grid_h) * p
            
            img_pos = torch.stack([
                torch.full((img_len,), img_start, device=device), 
                h_coords, 
                w_coords
            ], dim=-1).unsqueeze(0) 

            pos_ids = torch.cat([text_pos_3d, img_pos], dim=1) 
            cos, sin = self.rope(pos_ids) 
            self.rope_cache[cache_key] = (cos, sin)
            
        cos, sin = self.rope_cache[cache_key]
        return cos.expand(B, -1, -1), sin.expand(B, -1, -1)

    def freeze_base_for_lora(self):
        for name, param in self.named_parameters():
            if "final_layers" in name or "cap_embedder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.dirac_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

        for block in self.noise_refiner + self.blocks:
            nn.init.constant_(block.adaLN_msa[-1].weight, 0)
            nn.init.constant_(block.adaLN_msa[-1].bias, 0)
            nn.init.constant_(block.adaLN_mlp[-1].weight, 0)
            nn.init.constant_(block.adaLN_mlp[-1].bias, 0)
            
        for block in self.context_refiner:
            nn.init.constant_(block.attn_out.weight, 0)
            nn.init.constant_(block.ffn.w3.weight, 0)

        for ps in self.final_layers.keys():
            nn.init.zeros_(self.final_layers[ps].weight)
            nn.init.zeros_(self.final_layers[ps].bias)

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        text_embeds: torch.Tensor, 
        text_mask: Optional[torch.Tensor] = None, 
        patch_size: Optional[int] = None,
        gradient_checkpointing: bool = Config.gradient_checkpointing,
        has_padding: Optional[bool] = None 
    ) -> torch.Tensor:
        if patch_size is None:
            patch_size = self.patch_size

        B, C, H, W = x.shape
        p = patch_size
        grid_h, grid_w = H // p, W // p
        img_len = grid_h * grid_w

        # Step 1: Patchify & Embed Image
        x_patches = x.reshape(B, C, grid_h, p, grid_w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, img_len, -1)
        x_embedded = self.x_embedders[str(p)](x_patches)

        # Step 2: Text Dropouts for CFG support
        if self.training and self.cond_dropout_prob > 0.0:
            dropout_mask = torch.rand(B, 1, 1, device=x.device) > self.cond_dropout_prob
            text_embeds = text_embeds * dropout_mask

        # Step 3: Text Processing
        context = self.cap_embedder(text_embeds)
        seq_len_text = context.shape[1]
        
        if text_mask is None:
            text_mask = torch.ones(B, seq_len_text, dtype=torch.bool, device=x.device)

        is_null = (text_embeds.abs().sum(dim=(1, 2)) == 0).view(B, 1, 1)
        context = torch.where(is_null, self.cap_pad_token.expand_as(context), context)

        # Step 4: Timestep Embedding
        t_emb = self.t_embedder(t)

        # Step 5: High-Performance Positional Embedding Cache Lookup
        rope_freqs_all = self.get_rope_freqs(seq_len_text, grid_h, grid_w, x.device, p, B)

        rope_img = (rope_freqs_all[0][:, seq_len_text:, :], rope_freqs_all[1][:, seq_len_text:, :])
        rope_txt = (rope_freqs_all[0][:, :seq_len_text, :], rope_freqs_all[1][:, :seq_len_text, :])

        if has_padding is None:
            has_padding = text_mask is not None and not bool(text_mask.all().item())

        # Step 6: Noise Refiner (Image Only)
        for block in self.noise_refiner:
            if gradient_checkpointing:
                def make_noise_forward(b, c_val, h_val, w_val, r_val):
                    def noise_forward(x_in):
                        return b(x_in, c_val, None, 0, h_val, w_val, r_val, False, True)
                    return noise_forward
                x_embedded = checkpoint(
                    make_noise_forward(block, t_emb, grid_h, grid_w, rope_img), x_embedded, use_reentrant=False
                )
            else:
                x_embedded = block(
                    x_embedded, t_emb, None, 0, grid_h, grid_w, rope_img, False, True
                )

        # Step 7: Context Refiner (Text Only)
        for block in self.context_refiner:
            if gradient_checkpointing:
                def make_context_forward(b, m_val, s_val, r_val, p_val):
                    def context_forward(x_in):
                        return b(x_in, None, m_val, s_val, 0, 0, r_val, p_val, False)
                    return context_forward
                context = checkpoint(
                    make_context_forward(block, text_mask, seq_len_text, rope_txt, has_padding), context, use_reentrant=False
                )
            else:
                context = block(
                    context, None, text_mask, seq_len_text, 0, 0, rope_txt, has_padding, False
                )

        # Step 8: Unified Joint Stream
        unified = torch.cat([context, x_embedded], dim=1)
        joint_mask = torch.cat([text_mask, torch.ones(B, img_len, dtype=torch.bool, device=x.device)], dim=1)

        for block in self.blocks:
            if gradient_checkpointing:
                def make_joint_forward(b, c_val, m_val, s_val, h_val, w_val, r_val, p_val):
                    def joint_forward(x_in):
                        return b(x_in, c_val, m_val, s_val, h_val, w_val, r_val, p_val, True)
                    return joint_forward
                unified = checkpoint(
                    make_joint_forward(block, t_emb, joint_mask, seq_len_text, grid_h, grid_w, rope_freqs_all, has_padding), unified, use_reentrant=False
                )
            else:
                unified = block(
                    unified, t_emb, joint_mask, seq_len_text, grid_h, grid_w, rope_freqs_all, has_padding, True
                )

        # Step 9: Unpatchify & Output Generation
        x_out = unified[:, -img_len:, :]
        x_out = self.final_norm(x_out)
        x_out = self.final_layers[str(p)](x_out)

        x_out = x_out.view(B, grid_h, grid_w, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x_out