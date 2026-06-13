from typing import Optional

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.PatchTST_backbone import PatchTST_backbone
from models.DLinear import series_decomp
from utils.timefeatures import time_features_from_frequency_str


class HamiltonianMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualPatchTST(nn.Module):
    def __init__(
        self,
        d_in: int,
        time_dim: int,
        d_out: int,
        input_len: int,
        target_window: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.05,
        fc_dropout: float = 0.05,
        head_dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: Optional[str] = "end",
        attn_dropout: float = 0.0,
        norm: str = "BatchNorm",
        activation: str = "gelu",
        key_padding_mask: bool = "auto",
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        bound_output: bool = True,
        init_u_scale: float = 1.0,
        mem_debug: bool = False,
    ):
        super().__init__()
        self.time_dim = int(time_dim)
        self.d_out = int(d_out)
        self.target_window = int(target_window)
        c_in = d_in + self.time_dim
        self.mem_debug = bool(mem_debug)
        self._mem_debug_printed = False

        self.backbone = PatchTST_backbone(
            c_in=c_in,
            context_window=input_len,
            target_window=self.target_window,
            patch_len=patch_len,
            stride=stride,
            n_layers=e_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=activation,
            key_padding_mask=key_padding_mask,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=False,
            head_type="flatten",
            individual=False,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
        )

        self.bound_output = bool(bound_output)
        if self.bound_output:
            self.u_scale = nn.Parameter(torch.tensor(float(init_u_scale)))
        else:
            self.register_parameter("u_scale", None)

    def forward(self, x_hist, x_mark):
        mem_debug = self.mem_debug and torch.cuda.is_available()
        mem_device = x_hist.device if mem_debug else None
        if mem_debug and not self._mem_debug_printed:
            torch.cuda.reset_peak_memory_stats(mem_device)

        if (
            self.time_dim > 0
            and (
                x_mark is None
                or x_mark.dim() != 3
                or x_mark.size(1) != x_hist.size(1)
                or x_mark.size(2) != self.time_dim
            )
        ):
            x_mark = torch.zeros(
                x_hist.size(0),
                x_hist.size(1),
                self.time_dim,
                device=x_hist.device,
                dtype=x_hist.dtype,
            )

        if self.time_dim > 0:
            x_in = torch.cat([x_hist, x_mark], dim=-1)
        else:
            x_in = x_hist

        pred = self.backbone(x_in.permute(0, 2, 1))
        pred = pred.permute(0, 2, 1)[..., : self.d_out]
        if self.target_window == 1:
            pred = pred[:, -1, :]

        if self.bound_output:
            pred = torch.tanh(pred) * self.u_scale

        if mem_debug and not self._mem_debug_printed:
            peak = torch.cuda.max_memory_allocated(mem_device)
            x_in_bytes = x_in.numel() * x_in.element_size()
            pred_bytes = pred.numel() * pred.element_size()
            print(
                "[mem_debug][ResidualPatchTST] x_in=%.2fMB pred=%.2fMB peak=%.2fMB"
                % (x_in_bytes / 1e6, pred_bytes / 1e6, peak / 1e6)
            )
            print(torch.cuda.memory_summary(mem_device, abbreviated=True))
            self._mem_debug_printed = True
        return pred


class ResidualDLinear(nn.Module):
    def __init__(
        self,
        d_in: int,
        time_dim: int,
        d_out: int,
        input_len: int,
        target_window: int = 1,
        kernel_size: int = 25,
        individual: bool = False,
        bound_output: bool = True,
        init_u_scale: float = 1.0,
    ):
        super().__init__()
        self.time_dim = int(time_dim)
        self.d_out = int(d_out)
        self.target_window = int(target_window)
        self.individual = bool(individual)
        self.c_in = d_in + self.time_dim

        self.decompsition = series_decomp(int(kernel_size))
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.c_in):
                self.Linear_Seasonal.append(nn.Linear(input_len, self.target_window))
                self.Linear_Trend.append(nn.Linear(input_len, self.target_window))
        else:
            self.Linear_Seasonal = nn.Linear(input_len, self.target_window)
            self.Linear_Trend = nn.Linear(input_len, self.target_window)

        self.bound_output = bool(bound_output)
        if not self.bound_output:
            self.u_scale = nn.Parameter(torch.tensor(float(init_u_scale)))
        else:
            self.register_parameter("u_scale", None)

    def forward(self, x_hist, x_mark):
        if (
            self.time_dim > 0
            and (
                x_mark is None
                or x_mark.dim() != 3
                or x_mark.size(1) != x_hist.size(1)
                or x_mark.size(2) != self.time_dim
            )
        ):
            x_mark = torch.zeros(
                x_hist.size(0),
                x_hist.size(1),
                self.time_dim,
                device=x_hist.device,
                dtype=x_hist.dtype,
            )

        if self.time_dim > 0:
            x_in = torch.cat([x_hist, x_mark], dim=-1)
        else:
            x_in = x_hist

        seasonal_init, trend_init = self.decompsition(x_in)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            batch = seasonal_init.size(0)
            seasonal_output = torch.zeros(
                batch, self.c_in, self.target_window, device=x_in.device, dtype=x_in.dtype
            )
            trend_output = torch.zeros(
                batch, self.c_in, self.target_window, device=x_in.device, dtype=x_in.dtype
            )
            for i in range(self.c_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        pred = seasonal_output + trend_output
        pred = pred.permute(0, 2, 1)[..., : self.d_out]
        if self.target_window == 1:
            pred = pred[:, -1, :]

        if not self.bound_output:
            pred = torch.tanh(pred) * self.u_scale
        return pred * 0


class PHVectorField(nn.Module):
    def __init__(self, d: int, rank: int = 4):
        super().__init__()
        self.d = d
        self.A = nn.Parameter(0.01 * torch.randn(d, d))
        self.B = nn.Parameter(0.01 * torch.randn(d, rank))

    def forward(self, gradH):
        J = self.A - self.A.t()
        R = self.B @ self.B.t()
        M = J - R
        return gradH @ M.t()


class DiffusionFactorMLP(nn.Module):
    def __init__(self, d: int, rank: int = 4, hidden: int = 64):
        super().__init__()
        self.d = d
        self.rank = rank
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d * rank),
        )

    def forward(self, x):
        batch = x.size(0)
        out = self.net(x).view(batch, self.d, self.rank)
        return out


def batch_outer(vec: torch.Tensor) -> torch.Tensor:
    return vec.unsqueeze(2) * vec.unsqueeze(1)


class PHEnergyAttentionPatchTSTMulti(nn.Module):
    def __init__(
        self,
        d: int = 5,
        c: int = 10,
        input_len: int = 36,
        K: int = 12,
        dt: float = 0.7,
        cov_rollout: str = "last",
        hidden_H: int = 80,
        pt_d_model: int = 128,
        pt_nlayers: int = 2,
        pt_nhead: int = 8,
        pt_d_ff: int = 256,
        pt_dropout: float = 0.05,
        pt_fc_dropout: float = 0.05,
        pt_head_dropout: float = 0.0,
        pt_patch_len: int = 16,
        pt_stride: int = 8,
        pt_padding_patch: Optional[str] = "end",
        pt_attn_dropout: float = 0.0,
        pt_norm: str = "BatchNorm",
        pt_activation: str = "gelu",
        pt_key_padding_mask: bool = "auto",
        pt_pre_norm: bool = False,
        pt_store_attn: bool = False,
        pt_pe: str = "zeros",
        pt_learn_pe: bool = True,
        pt_revin: bool = True,
        pt_affine: bool = True,
        pt_subtract_last: bool = False,
        use_hamiltonian: bool = True,
        use_attention: bool = True,
        use_diffusion: bool = True,
        use_dlinear_residual: bool = True,
        use_covariate: bool = False,
        mem_debug: bool = False,
        diff_rank: int = 4,
        diff_hidden: int = 64,
        diff_scale: float = 1.0,
        midpoint_iters: int = 4,
        midpoint_tol: float = 0.0,
        bound_output: bool = True,
        init_u_scale: float = 1.0,
        dlinear_kernel_size: int = 25,
        dlinear_individual: bool = False,
    ):
        super().__init__()
        self.d, self.c = d, c
        self.input_len = input_len
        self.K = K
        self.dt = float(dt)
        self.cov_rollout = cov_rollout

        self.use_hamiltonian = bool(use_hamiltonian)
        self.use_attention = bool(use_attention)
        self.use_diffusion = bool(use_diffusion)
        self.use_dlinear_residual = bool(use_dlinear_residual)
        self.use_covariate = bool(use_covariate)

        self.diff_rank = min(int(diff_rank), d)
        self.diff_scale = float(diff_scale)
        self.midpoint_iters = int(midpoint_iters)
        self.midpoint_tol = float(midpoint_tol)

        self.H = HamiltonianMLP(d=d, hidden=hidden_H)
        self.ph = PHVectorField(d=d, rank=min(4, d))
        self.residual_dlinear = ResidualDLinear(
            d_in=d,
            time_dim=c,
            d_out=d,
            input_len=input_len,
            target_window=K,
            kernel_size=dlinear_kernel_size,
            individual=dlinear_individual,
            bound_output=bound_output,
            init_u_scale=init_u_scale,
        )
        self.residual_patchtst = ResidualPatchTST(
            d_in=d,
            time_dim=c,
            d_out=d,
            input_len=input_len,
            target_window=K,
            d_model=pt_d_model,
            n_heads=pt_nhead,
            e_layers=pt_nlayers,
            d_ff=pt_d_ff,
            dropout=pt_dropout,
            fc_dropout=pt_fc_dropout,
            head_dropout=pt_head_dropout,
            patch_len=pt_patch_len,
            stride=pt_stride,
            padding_patch=pt_padding_patch,
            attn_dropout=pt_attn_dropout,
            norm=pt_norm,
            activation=pt_activation,
            key_padding_mask=pt_key_padding_mask,
            pre_norm=pt_pre_norm,
            store_attn=pt_store_attn,
            pe=pt_pe,
            learn_pe=pt_learn_pe,
            revin=pt_revin,
            affine=pt_affine,
            subtract_last=pt_subtract_last,
            bound_output=bound_output,
            init_u_scale=init_u_scale,
            mem_debug=mem_debug,
        )
        self.S = DiffusionFactorMLP(d=d, rank=self.diff_rank, hidden=diff_hidden)

    def _compute_energy_aux_from_forecast(self, x0, y_hat, return_gradH: bool = True):
        if not self.use_hamiltonian:
            return {}

        x_seq = torch.cat([x0.unsqueeze(1), y_hat], dim=1)
        b, t, d = x_seq.shape

        if return_gradH:
            with torch.enable_grad():
                x_req = x_seq.detach().clone().requires_grad_(True)
                Hx = self.H(x_req.reshape(b * t, d)).reshape(b, t)
                gradHx = torch.autograd.grad(Hx.sum(), x_req, create_graph=True)[0]
            return {
                "H_t": Hx[:, :-1],
                "H_next": Hx[:, 1:],
                "gradH_t": gradHx[:, :-1],
                "gradH_next": gradHx[:, 1:],
            }

        Hx = self.H(x_seq.reshape(b * t, d)).reshape(b, t)
        return {"H_t": Hx[:, :-1], "H_next": Hx[:, 1:]}

    def _build_cov_full(self, cov_hist, steps: int):
        if cov_hist is None:
            return None
        cov_full = cov_hist
        for _ in range(int(steps)):
            cov_full = self._extend_cov(cov_full)
        return cov_full

    def _extend_cov(self, cov_hist):
        if cov_hist is None:
            return None

        if self.cov_rollout == "last":
            cov_next = cov_hist[:, -1:, :]
        elif self.cov_rollout == "zero":
            cov_next = torch.zeros_like(cov_hist[:, -1:, :])
        else:
            raise ValueError("cov_rollout must be 'last' or 'zero'")

        return torch.cat([cov_hist, cov_next], dim=1)

    def _make_seq(self, x_hist, cov_hist):
        if cov_hist is None:
            zeros = torch.zeros(
                x_hist.size(0),
                x_hist.size(1),
                self.c,
                device=x_hist.device,
                dtype=x_hist.dtype,
            )
            return torch.cat([x_hist, zeros], dim=-1)
        return torch.cat([x_hist, cov_hist], dim=-1)

    def _gradH(self, x):
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            Hx = self.H(x_req)
            gradHx = torch.autograd.grad(Hx.sum(), x_req, create_graph=True)[0]
        return Hx, gradHx

    def _H_only(self, x):
        return self.H(x)

    def _drift_and_diffusion(self, x_eval, x_hist, cov_hist, y_hat=None, step=None):
        if self.use_hamiltonian:
            _, gradH = self._gradH(x_eval)
            dx_phys = self.ph(gradH)
        else:
            dx_phys = torch.zeros_like(x_eval)

        if self.use_attention:
            attn_pred = None
            if y_hat is not None and step is not None:
                attn_pred = y_hat[:, step, :]
            if attn_pred is None:
                attn_pred = x_eval
            dx_attn = attn_pred - x_eval
        else:
            dx_attn = torch.zeros_like(x_eval)

        drift = dx_phys + dx_attn
        if self.use_diffusion:
            S_eval = self.diff_scale * self.S(x_eval)
        else:
            S_eval = torch.zeros(
                x_eval.size(0),
                self.d,
                self.diff_rank,
                device=x_eval.device,
                dtype=x_eval.dtype,
            )
        return drift, S_eval

    def forward(
        self,
        past_target,
        past_covariates=None,
        future_covariates=None,
        return_aux: bool = True,
        return_gradH: bool = True,
    ):
        del future_covariates
        param_dtype = next(self.parameters()).dtype
        x_hist = past_target.to(dtype=param_dtype)
        cov_hist = None
        if self.use_covariate and past_covariates is not None:
            cov_hist = past_covariates.to(dtype=param_dtype)

        dlinear_hat = None
        if self.use_dlinear_residual:
            dlinear_hat = self.residual_dlinear(x_hist, cov_hist)
        y_hat = self.residual_patchtst(x_hist, cov_hist)
        if dlinear_hat is not None:
            y_hat = y_hat + dlinear_hat
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)
        if dlinear_hat is not None and dlinear_hat.dim() == 2:
            dlinear_hat = dlinear_hat.unsqueeze(1)
        if not return_aux:
            return y_hat, {}

        x0 = x_hist[:, -1, :]
        aux = self._compute_energy_aux_from_forecast(x0, y_hat, return_gradH=return_gradH)
        return y_hat, aux


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = True

        self.d = configs.enc_in
        self.c = self._infer_time_feat_dim(configs)

        self.dt = float(getattr(configs, "pha_dt", 0.7))
        self.cov_rollout = getattr(configs, "pha_cov_rollout", "last")
        self.lambda_energy = float(getattr(configs, "pha_lambda_energy", 0.05))
        self.energy_mode = getattr(configs, "pha_energy_mode", "H_diff")
        self.lambda_struct = float(getattr(configs, "pha_lambda_struct", 1e-4))
        self.lambda_moment = float(getattr(configs, "pha_lambda_moment", 1.0))
        self.lambda_diff_reg = float(getattr(configs, "pha_lambda_diff_reg", 1e-4))
        self.loss_weighting = str(getattr(configs, "pha_loss_weighting", "fixed"))
        self.loss_ema_beta = float(getattr(configs, "pha_loss_ema_beta", 0.99))
        self.loss_ema_eps = float(getattr(configs, "pha_loss_ema_eps", 1e-6))
        self.loss_weight_min = float(getattr(configs, "pha_loss_weight_min", 1e-2))
        self.loss_weight_max = float(getattr(configs, "pha_loss_weight_max", 3.0))

        self.use_hamiltonian = bool(getattr(configs, "pha_use_hamiltonian", True))
        self.use_attention = bool(getattr(configs, "pha_use_attention", True))
        self.use_diffusion = bool(getattr(configs, "pha_use_diffusion", True))
        self.use_dlinear_residual = bool(getattr(configs, "pha_use_dlinear_residual", True))
        self.use_covariate = bool(getattr(configs, "pha_use_covariate", False))
        if not self.use_covariate:
            self.c = 0
        self.mem_debug = bool(getattr(configs, "pha_mem_debug", False))

        self.diff_rank = int(getattr(configs, "pha_diff_rank", 4))
        self.diff_hidden = int(getattr(configs, "pha_diff_hidden", 64))
        self.diff_scale = float(getattr(configs, "pha_diff_scale", 1.0))
        self.midpoint_iters = int(getattr(configs, "pha_midpoint_iters", 4))
        self.midpoint_tol = float(getattr(configs, "pha_midpoint_tol", 0.0))

        self.hidden_H = int(getattr(configs, "pha_hidden_H", 96))

        self.pt_d_model = int(getattr(configs, "pha_pt_d_model", getattr(configs, "d_model", 128)))
        self.pt_nlayers = int(getattr(configs, "pha_pt_nlayers", getattr(configs, "e_layers", 2)))
        self.pt_nhead = int(getattr(configs, "pha_pt_nhead", getattr(configs, "n_heads", 8)))
        self.pt_d_ff = int(getattr(configs, "pha_pt_d_ff", getattr(configs, "d_ff", 256)))
        self.pt_dropout = float(getattr(configs, "pha_pt_dropout", getattr(configs, "dropout", 0.05)))
        self.pt_fc_dropout = float(
            getattr(configs, "pha_pt_fc_dropout", getattr(configs, "fc_dropout", 0.05))
        )
        self.pt_head_dropout = float(
            getattr(configs, "pha_pt_head_dropout", getattr(configs, "head_dropout", 0.0))
        )
        self.pt_patch_len = int(getattr(configs, "pha_pt_patch_len", getattr(configs, "patch_len", 16)))
        self.pt_stride = int(getattr(configs, "pha_pt_stride", getattr(configs, "stride", 8)))
        self.pt_padding_patch = getattr(
            configs, "pha_pt_padding_patch", getattr(configs, "padding_patch", "end")
        )
        self.pt_attn_dropout = float(getattr(configs, "pha_pt_attn_dropout", 0.0))
        self.pt_norm = str(getattr(configs, "pha_pt_norm", "BatchNorm"))
        self.pt_activation = str(getattr(configs, "pha_pt_activation", "gelu"))
        self.pt_key_padding_mask = getattr(configs, "pha_pt_key_padding_mask", "auto")
        self.pt_pre_norm = bool(getattr(configs, "pha_pt_pre_norm", False))
        self.pt_store_attn = bool(getattr(configs, "pha_pt_store_attn", False))
        self.pt_pe = str(getattr(configs, "pha_pt_pe", "zeros"))
        self.pt_learn_pe = bool(getattr(configs, "pha_pt_learn_pe", True))
        self.pt_revin = bool(getattr(configs, "pha_pt_revin", getattr(configs, "revin", 1)))
        self.pt_affine = bool(getattr(configs, "pha_pt_affine", getattr(configs, "affine", 0)))
        self.pt_subtract_last = bool(
            getattr(configs, "pha_pt_subtract_last", getattr(configs, "subtract_last", 0))
        )
        self.bound_output = bool(getattr(configs, "pha_bound_output", True))
        self.init_u_scale = float(getattr(configs, "pha_init_u_scale", 1.0))
        self.dlinear_kernel_size = int(getattr(configs, "pha_dlinear_kernel_size", 25))
        self.dlinear_individual = bool(getattr(configs, "pha_dlinear_individual", False))

        self.network = PHEnergyAttentionPatchTSTMulti(
            d=self.d,
            c=self.c,
            input_len=self.seq_len,
            K=self.pred_len,
            dt=self.dt,
            cov_rollout=self.cov_rollout,
            hidden_H=self.hidden_H,
            pt_d_model=self.pt_d_model,
            pt_nlayers=self.pt_nlayers,
            pt_nhead=self.pt_nhead,
            pt_d_ff=self.pt_d_ff,
            pt_dropout=self.pt_dropout,
            pt_fc_dropout=self.pt_fc_dropout,
            pt_head_dropout=self.pt_head_dropout,
            pt_patch_len=self.pt_patch_len,
            pt_stride=self.pt_stride,
            pt_padding_patch=self.pt_padding_patch,
            pt_attn_dropout=self.pt_attn_dropout,
            pt_norm=self.pt_norm,
            pt_activation=self.pt_activation,
            pt_key_padding_mask=self.pt_key_padding_mask,
            pt_pre_norm=self.pt_pre_norm,
            pt_store_attn=self.pt_store_attn,
            pt_pe=self.pt_pe,
            pt_learn_pe=self.pt_learn_pe,
            pt_revin=self.pt_revin,
            pt_affine=self.pt_affine,
            pt_subtract_last=self.pt_subtract_last,
            use_hamiltonian=self.use_hamiltonian,
            use_attention=self.use_attention,
            use_diffusion=self.use_diffusion,
            use_dlinear_residual=self.use_dlinear_residual,
            use_covariate=self.use_covariate,
            mem_debug=self.mem_debug,
            diff_rank=self.diff_rank,
            diff_hidden=self.diff_hidden,
            diff_scale=self.diff_scale,
            midpoint_iters=self.midpoint_iters,
            midpoint_tol=self.midpoint_tol,
            bound_output=self.bound_output,
            init_u_scale=self.init_u_scale,
            dlinear_kernel_size=self.dlinear_kernel_size,
            dlinear_individual=self.dlinear_individual,
        )

        if self.loss_weighting == "uncertainty":
            self.loss_log_vars = nn.ParameterDict(
                {
                    "mse": nn.Parameter(torch.zeros(())),
                    "energy": nn.Parameter(torch.zeros(())),
                    "struct": nn.Parameter(torch.zeros(())),
                    "moment": nn.Parameter(torch.zeros(())),
                    "diff_reg": nn.Parameter(torch.zeros(())),
                }
            )
        elif self.loss_weighting == "ema_norm":
            self.register_buffer("loss_ema_mse", torch.zeros(()))
            self.register_buffer("loss_ema_energy", torch.zeros(()))
            self.register_buffer("loss_ema_struct", torch.zeros(()))
            self.register_buffer("loss_ema_moment", torch.zeros(()))
            self.register_buffer("loss_ema_diff_reg", torch.zeros(()))

    def _infer_time_feat_dim(self, configs) -> int:
        if getattr(configs, "embed", "timeF") != "timeF":
            return 4
        freq = getattr(configs, "freq", "h")
        return len(time_features_from_frequency_str(freq))

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        return_aux = bool(self.training)
        y_hat, aux = self.network(
            x_enc,
            x_mark_enc,
            future_covariates=None,
            return_aux=return_aux,
            return_gradH=return_aux and self.energy_mode == "grad_diff",
        )
        return y_hat, aux

    def compute_loss(
        self,
        outputs,
        target,
        batch_x,
        batch_y_full,
        batch_x_mark,
        batch_y_mark=None,
        aux=None,
    ):
        if isinstance(outputs, tuple):
            y_hat, aux = outputs
        else:
            y_hat = outputs

        need_aux = aux is None
        if not need_aux and self.use_hamiltonian:
            if self.energy_mode == "H_diff":
                need_aux = "H_next" not in aux or "H_t" not in aux
            else:
                need_aux = "gradH_next" not in aux or "gradH_t" not in aux

        if need_aux:
            _, aux = self.network(
                batch_x,
                batch_x_mark,
                return_aux=True,
                return_gradH=self.energy_mode == "grad_diff",
            )

        target = target.to(dtype=y_hat.dtype)

        mse = F.mse_loss(y_hat, target)

        if self.use_hamiltonian:
            if self.energy_mode == "H_diff":
                e_loss = F.mse_loss(aux["H_next"], aux["H_t"].detach())
            elif self.energy_mode == "grad_diff":
                e_loss = F.mse_loss(aux["gradH_next"], aux["gradH_t"].detach())
            else:
                raise ValueError("energy_mode must be 'H_diff' or 'grad_diff'")
        else:
            e_loss = y_hat.new_zeros(())

        if self.use_hamiltonian:
            A = self.network.ph.A
            B = self.network.ph.B
            struct_loss = A.pow(2).mean() + B.pow(2).mean()
        else:
            struct_loss = y_hat.new_zeros(())

        aux_dtype = y_hat.dtype
        aux_device = y_hat.device

        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)

        dt = float(self.network.dt)
        sqrt_dt = dt ** 0.5
        K = int(self.pred_len)

        x_hist = batch_x[:, :, : self.d].to(device=aux_device, dtype=aux_dtype)
        y_true = batch_y_full[
            :, self.label_len : self.label_len + K, : self.d
        ].to(device=aux_device, dtype=aux_dtype)
        x_full = torch.cat([x_hist, y_true], dim=1)

        cov_hist = None
        if self.use_covariate:
            cov_hist = batch_x_mark.to(device=aux_device, dtype=aux_dtype)

        cov_full = None
        if self.use_covariate:
            if batch_y_mark is not None:
                cov_future = batch_y_mark[:, self.label_len : self.label_len + K, :].to(
                    device=aux_device, dtype=aux_dtype
                )
                if cov_future.size(1) < K:
                    pad = cov_future[:, -1:, :].expand(-1, K - cov_future.size(1), -1)
                    cov_future = torch.cat([cov_future, pad], dim=1)
                cov_full = torch.cat([cov_hist, cov_future[:, :K, :]], dim=1)
            else:
                cov_full = self.network._build_cov_full(cov_hist, K)

        drift_losses = []
        diff_losses = []
        diff_regs = []

        for step in range(K):
            x_hist_step = x_full[:, step : step + self.seq_len, :]
            x_t = x_hist_step[:, -1, :]
            x_true_next = x_full[:, step + self.seq_len, :]
            delta_x = x_true_next - x_t

            cov_hist_step = None
            if cov_full is not None:
                cov_hist_step = cov_full[:, step : step + self.seq_len, :]

            drift_step, S_step = self.network._drift_and_diffusion(
                x_t,
                x_hist_step,
                cov_hist_step,
                y_hat=y_hat,
                step=step,
            )
            drift_losses.append(F.mse_loss(delta_x, dt * drift_step))

            if self.use_diffusion:
                r = (delta_x - dt * drift_step) / (sqrt_dt + 1e-8)
                rrT = batch_outer(r)
                Sigma = S_step @ S_step.transpose(1, 2)
                diff_losses.append(F.mse_loss(rrT, Sigma))
            else:
                diff_losses.append(y_hat.new_zeros(()))

            diff_regs.append(S_step.pow(2).mean() if self.use_diffusion else y_hat.new_zeros(()))

        drift_loss = torch.stack(drift_losses).mean()
        diff_loss = torch.stack(diff_losses).mean()
        diff_reg = torch.stack(diff_regs).mean()

        moment_loss_raw = drift_loss + 0.1 * diff_loss
        moment_loss = torch.log1p(moment_loss_raw).clamp(max=10.0)

        loss_terms = {
            "mse": mse,
            "energy": e_loss,
            "struct": struct_loss,
            "moment": moment_loss,
            "diff_reg": diff_reg,
        }
        base_lambdas = {
            "mse": 1.0,
            "energy": self.lambda_energy,
            "struct": self.lambda_struct,
            "moment": self.lambda_moment,
            "diff_reg": self.lambda_diff_reg,
        }

        if self.loss_weighting == "uncertainty":
            total = mse.new_zeros(())
            effective_weights = {}
            for name, term in loss_terms.items():
                scaled = base_lambdas[name] * term
                log_var = self.loss_log_vars[name]
                weight = 0.5 * torch.exp(-log_var)
                total = total + weight * scaled + 0.5 * log_var
                effective_weights[name] = (weight * base_lambdas[name]).detach()
            loss = total
        elif self.loss_weighting == "ema_norm":
            if self.training:
                for name, term in loss_terms.items():
                    ema_name = f"loss_ema_{name}"
                    ema = getattr(self, ema_name)
                    if ema.item() == 0.0:
                        ema = term.detach()
                    else:
                        ema = ema * self.loss_ema_beta + term.detach() * (1.0 - self.loss_ema_beta)
                    setattr(self, ema_name, ema)
            ema_values = torch.stack(
                [
                    getattr(self, "loss_ema_mse"),
                    getattr(self, "loss_ema_energy"),
                    getattr(self, "loss_ema_struct"),
                    getattr(self, "loss_ema_moment"),
                    getattr(self, "loss_ema_diff_reg"),
                ]
            )
            ema_mean = ema_values.mean().clamp_min(self.loss_ema_eps)
            total = mse.new_zeros(())
            effective_weights = {}
            for name, term in loss_terms.items():
                ema = getattr(self, f"loss_ema_{name}")
                denom = (ema + self.loss_ema_eps).detach()
                weight = base_lambdas[name] * (ema_mean / denom)
                weight = weight.clamp(min=self.loss_weight_min, max=self.loss_weight_max)
                total = total + weight * term
                effective_weights[name] = weight.detach()
            loss = total
        else:
            loss = (
                mse
                + self.lambda_energy * e_loss
                + self.lambda_struct * struct_loss
                + self.lambda_moment * moment_loss
                + self.lambda_diff_reg * diff_reg
            )
            effective_weights = {
                "mse": torch.tensor(1.0, device=mse.device),
                "energy": torch.tensor(self.lambda_energy, device=mse.device),
                "struct": torch.tensor(self.lambda_struct, device=mse.device),
                "moment": torch.tensor(self.lambda_moment, device=mse.device),
                "diff_reg": torch.tensor(self.lambda_diff_reg, device=mse.device),
            }
        if True and random.random() < 0.05:
            print(
                "loss_weights_and_terms: "
                f"mse={effective_weights['mse'].item():.6f}*{mse.item():.6f}, "
                f"energy={effective_weights['energy'].item():.6f}*{e_loss.item():.6f}, "
                f"struct={effective_weights['struct'].item():.6f}*{struct_loss.item():.6f}, "
                f"moment={effective_weights['moment'].item():.6f}*{moment_loss.item():.6f}, "
                f"diff_reg={effective_weights['diff_reg'].item():.6f}*{diff_reg.item():.6f}"
            )
        return loss
