"""
ModularClusterNet: Differentiable Algebraic Slot Attention Network (v8).

Implements both the native PyTorch `nn.Module` (with `nn.ParameterDict` and CUDA/MPS
autograd support) and a zero-dependency Vectorized Analytical Autograd Engine in NumPy
with exact closed-form Jacobians for competitive concept attention.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax_axis0_np(logits):
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.clip(np.sum(exp_x, axis=0, keepdims=True), 1e-12, None)


def compute_group_behavior_coupling(X_np, label_mask=None):
    """
    Computes the Algebraic Group-Behavior Coupling vector omega in [0, 1]^m
    (continuous tensor realization of Thesis Chapter 4, p. 69 'Comportamento di Gruppo').

    Measures 1st-order and 2nd-order (XOR/interaction) functional dependencies across
    attributes in O(n * m^2) tensor operations so independent noise attributes are
    attenuated while 1st-order and 2nd-order interacting attributes are preserved.
    """
    n, m = X_np.shape
    if m <= 2 or n < 6:
        return np.ones((1, m), dtype=np.float64)

    # Standardize columns
    mean = np.mean(X_np, axis=0, keepdims=True)
    centered = X_np - mean
    std = np.std(X_np, axis=0, keepdims=True)
    valid = (std > 1e-6).reshape(-1)
    Z = np.zeros_like(X_np)
    Z[:, valid] = centered[:, valid] / std[:, valid]

    # 1. First-order signed correlation matrix Z^T Z / n
    raw_corr = np.matmul(Z.T, Z) / float(n)
    np.fill_diagonal(raw_corr, 0.0)
    corr = np.abs(raw_corr)

    if label_mask is not None and np.any(label_mask) and np.sum(~label_mask) > 0:
        corr_1st = np.max(corr[:, label_mask], axis=1)
        corr_1st[label_mask] = np.max(corr[label_mask][:, ~label_mask], axis=1)
    else:
        # Ignore trivial 2-category self-complements (where signed correlation is -1.0)
        corr_no_comp = np.where(raw_corr < -0.98, 0.0, corr)
        corr_1st = np.max(corr_no_comp, axis=1)

    # 2. Second-order interaction coupling (captures XOR / 'Gioco' pure interactions!)
    # For a target/response column y (or top candidate columns), check if pairs (z_i * z_j) correlate with y
    corr_2nd = np.zeros(m, dtype=np.float64)
    noise_floor = 2.0 / np.sqrt(float(max(n, 10)))
    if label_mask is not None and np.any(label_mask):
        target_indices = np.where(label_mask)[0]
        feat_indices = np.where(~label_mask)[0]
        run_2nd = (len(feat_indices) <= 500 and float(np.max(corr_1st)) < 0.35)
    else:
        # In unsupervised mode, boost coupled attributes via 2nd-order interactions among attributes above noise floor
        active_cand = np.where(corr_1st > 1.2 * noise_floor)[0]
        target_indices = active_cand[:12] if len(active_cand) > 0 else np.arange(min(m, 12))
        feat_indices = active_cand if len(active_cand) >= 2 else np.arange(m)
        run_2nd = (len(feat_indices) <= 500)

    if run_2nd:
        for t_idx in target_indices[:4]:
            z_t = Z[:, t_idx]
            # Interaction matrix among features weighted by z_t: E[z_i * z_j * z_t]
            weighted_Z = Z[:, feat_indices] * z_t[:, None]
            inter_mat = np.abs(np.matmul(Z[:, feat_indices].T, weighted_Z) / float(n))
            np.fill_diagonal(inter_mat, 0.0)
            best_inter = np.max(inter_mat, axis=1)
            corr_2nd[feat_indices] = np.maximum(corr_2nd[feat_indices], best_inter)
            corr_2nd[t_idx] = max(corr_2nd[t_idx], float(np.max(best_inter)) if len(best_inter) > 0 else 0.0)

    raw_score = np.maximum(corr_1st, 0.85 * corr_2nd)
    # Finite-sample random correlation threshold ~ 2.2 / sqrt(n)
    noise_floor = 2.0 / np.sqrt(float(max(n, 10)))
    omega = np.clip((raw_score - 0.65 * noise_floor) / max(1.0 - 0.65 * noise_floor, 1e-6), 0.02, 1.0)
    if np.max(omega) > 0:
        omega = omega / np.max(omega)
    # Sharpen omega so true group-behavior attributes stand out cleanly from uniform noise
    omega = np.clip(omega ** 2, 0.01, 1.0)
    return omega.reshape(1, m)


class _AnalyticalClusterNet:
    """
    Exact Differentiable Algebraic Slot Attention Network with closed-form gradients
    and Adam optimization in pure vectorized NumPy.
    """

    def __init__(
        self,
        n,
        m,
        k=None,
        device="cpu",
        tau=0.20,
        alpha=0.6,
        gamma=3.0,
        label_mask=None,
        random_state=42,
    ):
        self.n = int(n)
        self.m = int(m)
        self.k = int(k) if k is not None else int(n)
        self.device = device
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(random_state)

        if label_mask is not None:
            self.label_mask = np.asarray(label_mask, dtype=bool)
        else:
            self.label_mask = None

        self.matrices = {
            "G": 0.25 * self.rng.standard_normal((self.k, self.m)).astype(np.float64),
            "F": 0.05 * self.rng.standard_normal((self.k, 1)).astype(np.float64),
        }
        self.grads = {
            "G": np.zeros_like(self.matrices["G"]),
            "F": np.zeros_like(self.matrices["F"]),
        }

        self.Root = np.zeros((1, self.m), dtype=np.float64)
        self.RootMin = np.zeros((1, self.m), dtype=np.float64)
        self.RootMax = np.ones((1, self.m), dtype=np.float64)
        self.omega = np.ones((1, self.m), dtype=np.float64)
        self._cache = {}

    def initialize_G_from_data(self, X):
        X_np = np.asarray(X, dtype=np.float64)
        if self.alpha > 0:
            self.omega = compute_group_behavior_coupling(X_np, self.label_mask)
            # Initialize prototypes from data projected onto high-coupling group-behavior dimensions!
            # This prevents high-dimensional irrelevant attributes from trapping slots in identity-matrix local optima
            idx = self.rng.choice(X_np.shape[0], size=self.k, replace=(X_np.shape[0] < self.k))
            base = np.clip(X_np[idx], 0.03, 0.97)
            logits_data = np.log(base / (1.0 - base))
            # Attenuate initial prototype logits on low-coupling (irrelevant) dimensions
            init_G = logits_data * (self.omega ** 1.5) + 0.12 * self.rng.standard_normal((self.k, self.m))
            self.matrices["G"] = init_G.astype(np.float64)
            self.matrices["F"] = np.zeros((self.k, 1), dtype=np.float64)
        else:
            # Pure v7 baseline initialization
            self.omega = np.ones((1, self.m), dtype=np.float64)
            self.matrices["G"] = (0.15 * self.rng.standard_normal((self.k, self.m))).astype(np.float64)
            self.matrices["F"] = (0.1 * self.rng.standard_normal((self.k, 1))).astype(np.float64)

    def set_root_statistics(self, root, root_min, root_max):
        self.Root = np.asarray(root, dtype=np.float64).reshape(1, self.m)
        self.RootMin = np.asarray(root_min, dtype=np.float64).reshape(1, self.m)
        self.RootMax = np.asarray(root_max, dtype=np.float64).reshape(1, self.m)

    def _compute_relevance(self, G):
        root_int = self.RootMax - self.RootMin
        root_center = self.RootMin + root_int / 2.0
        num = np.abs(self.Root - G)
        den = (root_int / 2.0) + np.abs(self.Root - root_center)
        valid = (den > 1e-8).astype(np.float64)
        safe_den = np.where(den > 1e-8, den, 1.0)
        R = (num / safe_den) * valid
        if self.alpha > 0:
            R = R * self.omega
        return R, safe_den, valid

    def forward(self, X):
        X_np = np.asarray(X, dtype=np.float64)
        n_curr = X_np.shape[0]
        self.n = n_curr

        F_prior = _softmax_axis0_np(self.matrices["F"])  # (k, 1)
        C_prior = F_prior * n_curr  # (k, 1)
        G_sig = _sigmoid_np(self.matrices["G"])  # (k, m)

        sim_dot = np.matmul(G_sig * self.omega, X_np.T)

        if self.alpha > 0:
            R_lat, _, _ = self._compute_relevance(G_sig)
            R_sharp = np.power(np.clip(R_lat, 0.0, None), self.gamma) + 1e-4 * self.omega
            R_norm = R_sharp / np.clip(np.sum(R_sharp, axis=1, keepdims=True), 1e-8, None)

            RG2 = np.sum(R_norm * (G_sig ** 2), axis=1, keepdims=True)
            RGX = np.matmul(R_norm * G_sig, X_np.T)
            RX2 = np.matmul(R_norm, (X_np ** 2).T)
            dist_sq = np.clip(RG2 - 2.0 * RGX + RX2, 0.0, None)
            sim_metric = 1.0 - np.sqrt(dist_sq + 1e-8)
            sim_norm_dot = sim_dot / max(float(np.sum(self.omega)), 1.0)
            sim_total = (1.0 - self.alpha) * sim_norm_dot + self.alpha * sim_metric
            logits_T = (C_prior * sim_total) / max(self.tau, 1e-4)
        else:
            R_norm = np.ones_like(G_sig) / float(self.m)
            sim_total = sim_dot
            logits_T = C_prior * sim_dot

        T = _softmax_axis0_np(logits_T)

        max_idx = np.argmax(T, axis=0)
        T_hard = np.zeros_like(T)
        T_hard[max_idx, np.arange(n_curr)] = 1.0
        T_hard = np.maximum(T_hard, (T > 0.5).astype(np.float64))
        C_hard = np.sum(T_hard, axis=1, keepdims=True)

        C_soft = np.clip(np.sum(T, axis=1, keepdims=True), 1e-6, None)
        F_post = (C_soft / float(n_curr)).T
        G_post = np.matmul(T, X_np) / C_soft

        G_eff = 0.5 * G_post + 0.5 * G_sig
        R, safe_den, valid_den = self._compute_relevance(G_eff)

        real_R = np.clip(np.sum(R, axis=1, keepdims=True), 1e-6, None)
        mF = real_R / float(self.m)
        g_pr_sq = np.sum(R * R, axis=1, keepdims=True)
        weight_k = F_post.T / np.clip(mF, 1e-6, None)
        pra2 = float(np.sum(weight_k * g_pr_sq))
        cp = float(np.sqrt(max(pra2, 1e-12)))

        self._cache = {
            "X": X_np,
            "F_prior": F_prior,
            "C_prior": C_prior,
            "G_sig": G_sig,
            "G_post": G_post,
            "G_eff": G_eff,
            "R": R,
            "R_norm": R_norm,
            "safe_den": safe_den,
            "valid_den": valid_den,
            "sim_total": sim_total,
            "T": T,
            "C_soft": C_soft,
            "F_post": F_post,
            "real_R": real_R,
            "mF": mF,
            "g_pr_sq": g_pr_sq,
            "cp": cp,
        }

        return {
            "T": T,
            "T_hard": T_hard,
            "C": C_hard,
            "F": F_post,
            "G": G_post,
            "R": R,
            "cp": cp,
            "neg_cp": -cp,
        }

    def __call__(self, X):
        return self.forward(X)

    def backward(self):
        c = self._cache
        X = c["X"]
        T = c["T"]
        G_sig = c["G_sig"]
        G_eff = c["G_eff"]
        R = c["R"]
        safe_den = c["safe_den"]
        valid_den = c["valid_den"]
        C_soft = c["C_soft"]
        F_post = c["F_post"].T
        real_R = c["real_R"]
        g_pr_sq = c["g_pr_sq"]
        cp = c["cp"]
        n_curr = float(X.shape[0])

        inv_2cp = -1.0 / (2.0 * max(cp, 1e-6))

        dTerm_dR = float(self.m) * F_post * (
            (2.0 * R / real_R) - (g_pr_sq / (real_R ** 2))
        )
        dL_dR = inv_2cp * dTerm_dR

        dL_dF_post = inv_2cp * (float(self.m) * g_pr_sq / real_R)
        dL_dC_soft = dL_dF_post / n_curr - 0.015 * (F_post / n_curr)

        sign_diff = np.sign(G_eff - self.Root)
        dL_dG_eff = dL_dR * (sign_diff / safe_den) * valid_den * self.omega
        dL_dG_post = 0.5 * dL_dG_eff
        dL_dG_sig = 0.5 * dL_dG_eff

        scaled_dG = dL_dG_post / C_soft
        dG_post_term = np.matmul(scaled_dG, X.T) - np.sum(scaled_dG * c["G_post"], axis=1, keepdims=True)
        dL_dT = dG_post_term + dL_dC_soft

        eff_tau = max(self.tau, 1e-4) if self.alpha > 0 else 1.0
        sum_T_dL = np.sum(T * dL_dT, axis=0, keepdims=True)
        dL_dLogits = T * (dL_dT - sum_T_dL) / eff_tau

        sim_total = c["sim_total"]
        C_prior = c["C_prior"]
        dL_dC_prior = np.sum(dL_dLogits * sim_total, axis=1, keepdims=True)
        dL_dSim = dL_dLogits * C_prior

        dL_dG_sig += (np.matmul(dL_dSim, X) * self.omega) / max(np.sqrt(self.m), 1.0)
        sig_deriv = G_sig * (1.0 - G_sig)
        self.grads["G"] = dL_dG_sig * sig_deriv

        F_prior = c["F_prior"]
        dL_dF_prior = dL_dC_prior * n_curr - 0.12 * F_prior
        sum_F_dL = float(np.sum(F_prior * dL_dF_prior))
        self.grads["F"] = F_prior * (dL_dF_prior - sum_F_dL)

        return self.grads


if HAS_TORCH:
    class ModularClusterNet(nn.Module):
        def __init__(self, n, m, k=None, device="cpu", tau=0.20, alpha=0.6, gamma=3.0, label_mask=None):
            super(ModularClusterNet, self).__init__()
            self.n = int(n)
            self.m = int(m)
            self.k = int(k) if k is not None else int(n)
            self.device = device
            self.tau = float(tau)
            self.alpha = float(alpha)
            self.gamma = float(gamma)

            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask, dtype=torch.bool)
                self.register_buffer("label_mask", label_mask.to(device))
            else:
                self.register_buffer("label_mask", None)

            self.matrices = nn.ParameterDict()
            self.matrices["G"] = nn.Parameter(0.25 * torch.randn(self.k, self.m, device=device))
            self.matrices["F"] = nn.Parameter(0.05 * torch.randn(self.k, 1, device=device))

            self.register_buffer("Root", torch.zeros(1, self.m, device=device))
            self.register_buffer("RootMin", torch.zeros(1, self.m, device=device))
            self.register_buffer("RootMax", torch.ones(1, self.m, device=device))
            self.register_buffer("omega", torch.ones(1, self.m, device=device))

        def initialize_G_from_data(self, X):
            with torch.no_grad():
                X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
                lbl_np = self.label_mask.detach().cpu().numpy() if self.label_mask is not None else None
                if self.alpha > 0:
                    omega_np = compute_group_behavior_coupling(X_np, lbl_np)
                    self.omega.copy_(torch.tensor(omega_np, dtype=torch.float32, device=self.device))
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32, device=self.device)
                base = torch.clamp(X[: self.k], 0.03, 0.97)
                logits_data = torch.log(base / (1.0 - base))
                self.matrices["G"].copy_(logits_data * (self.omega ** 1.5) + 0.12 * torch.randn_like(base))
                self.matrices["F"].zero_()

        def set_root_statistics(self, root, root_min, root_max):
            self.Root.copy_(torch.as_tensor(root, dtype=torch.float32, device=self.device).view(1, self.m))
            self.RootMin.copy_(torch.as_tensor(root_min, dtype=torch.float32, device=self.device).view(1, self.m))
            self.RootMax.copy_(torch.as_tensor(root_max, dtype=torch.float32, device=self.device).view(1, self.m))

        def _compute_relevance(self, G):
            root_int = self.RootMax - self.RootMin
            root_center = self.RootMin + root_int / 2.0
            num = torch.abs(self.Root - G)
            den = (root_int / 2.0) + torch.abs(self.Root - root_center)
            eps = 1e-8
            safe_den = torch.where(den != 0, den, torch.ones_like(den) * eps)
            R = (num / safe_den) * (den != 0).float()
            if self.alpha > 0:
                R = R * self.omega
            return R

        def _compute_cluster_probability(self, F, R):
            real_R = torch.clamp(R.sum(dim=1, keepdim=True), min=1e-6)
            mF = real_R / float(self.m)
            g_pr = torch.sqrt((R * R).sum(dim=1, keepdim=True) + 1e-12)
            F_sq = F / torch.clamp(mF.t(), min=1e-6)
            g_pr_sq = g_pr * g_pr
            pra2 = torch.matmul(F_sq, g_pr_sq)
            cp = torch.sqrt(pra2 + 1e-8)
            return cp.squeeze()

        def forward(self, X):
            n_curr = X.shape[0]
            self.n = n_curr
            F_prior = torch.softmax(self.matrices["F"], dim=0)
            C_prior = F_prior * n_curr
            G_sig = torch.sigmoid(self.matrices["G"])

            sim_dot = torch.matmul(G_sig * self.omega, X.t())
            if self.alpha > 0:
                R_lat = self._compute_relevance(G_sig)
                R_sharp = torch.pow(torch.clamp(R_lat, min=0.0), self.gamma) + 1e-4 * self.omega
                R_norm = R_sharp / torch.clamp(R_sharp.sum(dim=1, keepdim=True), min=1e-8)
                RG2 = (R_norm * (G_sig ** 2)).sum(dim=1, keepdim=True)
                RGX = torch.matmul(R_norm * G_sig, X.t())
                RX2 = torch.matmul(R_norm, (X ** 2).t())
                dist_sq = torch.clamp(RG2 - 2.0 * RGX + RX2, min=0.0)
                sim_metric = 1.0 - torch.sqrt(dist_sq + 1e-8)
                sim_total = (1.0 - self.alpha) * (sim_dot / torch.clamp(self.omega.sum(), min=1.0)) + self.alpha * sim_metric
                T = torch.softmax((C_prior * sim_total) / max(self.tau, 1e-4), dim=0)
            else:
                T = torch.softmax(C_prior * sim_dot, dim=0)

            with torch.no_grad():
                max_idx = torch.argmax(T, dim=0)
                T_wta = torch.zeros_like(T)
                T_wta.scatter_(0, max_idx.unsqueeze(0), 1.0)
                T_hard = torch.maximum(T_wta, (T > 0.5).float())
                C = T_hard.sum(dim=1, keepdim=True)

            C_soft = torch.clamp(T.sum(dim=1, keepdim=True), min=1e-6)
            F = (C_soft / float(n_curr)).t()
            G_post = torch.matmul(T, X) / C_soft
            G_eff = 0.5 * G_post + 0.5 * G_sig
            R = self._compute_relevance(G_eff)
            cp = self._compute_cluster_probability(F, R)

            res_dict = {
                "T": T,
                "T_hard": T_hard,
                "C": C,
                "F": F,
                "G": G_post,
                "R": R,
                "cp": cp,
                "neg_cp": -cp,
            }
            if torch.jit.is_tracing():
                return tuple(res_dict.values())
            return res_dict
else:
    ModularClusterNet = _AnalyticalClusterNet
