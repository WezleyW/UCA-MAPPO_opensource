import os

import numpy as np
import torch
import torch.nn.functional as F

import env_core


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


class CentralValueNet(torch.nn.Module):
    def __init__(self, total_state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


def compute_entropy(probs):
    dist = torch.distributions.Categorical(probs)
    return dist.entropy().mean().item()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class UAV_Policy:
    """
    UCA-MAPPO:
    在 MAPPO 的基础上，仅修改优势构造：
      A_i,t = s_i,t * A_global,t
      s_i,t = 1 + lambda_t * phi_i,t
      phi_i,t = (N * w_i,t - 1) / (N - 1), N > 1
      lambda_t = g_m(t) * g_p(t) * theta
    其中 w_i,t 来源于各 UAV 在该步“self+neighbors 联合 U 图下降量”的 softmax 归一化，
    并在时间轴上对贡献做 EMA 平滑，
    lambda_t 采用双门控：
      lambda_t = theta * g_m(t) * g_p(t)
      g_m(t) = clip(m_t / m_min, 0, 1), m_t = mean_i c_i,t
      g_p(t) = clip(p_eff(t) / p_min, 0, 1), p_eff(t) 为有效参与度:
        neff = 1 / sum_i (c_hat_i,t^2), c_hat_i,t = c_i,t / sum_j c_j,t
        p_eff = (neff - 1) / (N - 1) (N>1), 映射到 [0,1]
      关闭门控规则:
        m_min == 0 -> g_m(t) 恒为 1
        p_min == 0 -> g_p(t) 恒为 1
        m_min == 0 且 p_min == 0 -> lambda_t 恒为 theta（不做动态门控）
    并支持:
      1) 权重平滑 (rho): w <- (1-rho)/N + rho*w
    """

    @classmethod
    def from_config(
        cls, # @classmethod 的第一个参数不是 self，而是“类对象本身”。
        run_cfg,
        method,
        team_size,
        state_dim,
        action_dim,
        device,
    ):
        return cls(
            run_cfg=run_cfg,
            method=method,
            team_size=team_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

    def __init__(
        self,
        run_cfg,
        method,
        team_size,
        state_dim,
        action_dim,
        device,
    ):
        policy_cfg = run_cfg[method]
        hidden_dim = int(policy_cfg["hidden_dim"])
        actor_lr = float(policy_cfg["actor_lr"])
        critic_lr = float(policy_cfg["critic_lr"])
        lmbda = float(policy_cfg["lmbda"])
        eps = float(policy_cfg["eps"])
        gamma = float(policy_cfg["gamma"])
        target_kl = float(policy_cfg["target_kl"])
        beta0 = float(policy_cfg["beta0"])
        beta_decay = float(policy_cfg["beta_decay"])
        beta_min = float(policy_cfg["beta_min"])
        method_name = str(method)
        uca_theta = float(policy_cfg.get("uca_theta", policy_cfg.get("uca_delta", 0.2)))
        uca_weight_norm = str(policy_cfg.get("uca_weight_norm", "softmax")).strip().lower()
        uca_tau = float(policy_cfg.get("uca_tau", 0.1))
        uca_rho = float(policy_cfg.get("uca_rho", 0.2))
        uca_m_min = float(policy_cfg.get("uca_m_min", 0.05))
        uca_p_min = float(policy_cfg.get("uca_p_min", 0.30))
        uca_ema_decay = float(policy_cfg.get("uca_ema_decay", 0.8))
        scheduler_warmup_steps = int(policy_cfg.get("scheduler_warmup_steps", 0))
        scheduler_decay_steps = int(policy_cfg.get("scheduler_decay_steps", 0))
        scheduler_decay_rate = float(policy_cfg.get("scheduler_decay_rate", 1.0))

        self.team_size = int(team_size)
        self.state_dim = int(state_dim)
        self.gamma = float(gamma)
        self.lmbda = float(lmbda)
        self.eps = float(eps)
        self.device = device
        self.target_kl = float(target_kl)
        self.beta0 = float(beta0)
        self.beta_decay = float(beta_decay)
        self.beta_min = float(beta_min)
        self.method_name = method_name
        self.weights_subdir = f"{self.method_name}_weights"
        self.plot_metric_keys = [
            "Total_Reward_per_Episode",
            "Episode_Length_per_Episode",
            "Policy_Loss_per_Episode",
            "Value_Loss_per_Episode",
            "Entropy_per_Episode",
            "KL_per_Episode",
            "Clip_frac_per_Episode",
        ]
        self.extra_log_metrics = [
            "policy_loss",
            "value_loss",
            "entropy",
            "kl",
            "clip_frac",
            "alpha_t_mean",
            "alpha_t_std",
            "alpha_t_min",
            "alpha_t_max",
        ]
        self.extra_save_metrics = [
            "policy_loss",
            "value_loss",
            "entropy",
            "kl",
            "clip_frac",
            "alpha_t_mean",
            "alpha_t_std",
            "alpha_t_min",
            "alpha_t_max",
        ]

        self.actor_base_lr = float(actor_lr)
        self.critic_base_lr = float(critic_lr)
        self.scheduler_warmup_steps = max(int(scheduler_warmup_steps), 0)
        self.scheduler_decay_steps = max(int(scheduler_decay_steps), 0)
        # 最终衰减到 base_lr * scheduler_decay_rate
        self.scheduler_decay_rate = float(np.clip(scheduler_decay_rate, 0.0, 1.0))

        self.actor = PolicyNet(self.state_dim, int(hidden_dim), int(action_dim)).to(self.device)
        self.critic = CentralValueNet(self.team_size * self.state_dim, int(hidden_dim)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_base_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_base_lr)

        self.uca_theta = max(float(uca_theta), 0.0)
        self.uca_weight_norm = uca_weight_norm
        if self.uca_weight_norm not in {"softmax", "linear"}:
            raise ValueError(
                f"uca_weight_norm must be 'softmax' or 'linear', got: {self.uca_weight_norm}"
            )
        self.uca_tau = max(float(uca_tau), 1e-6)
        self.uca_rho = float(np.clip(uca_rho, 0.0, 1.0))
        self.uca_m_min = float(uca_m_min)
        self.uca_p_min = float(uca_p_min)
        self.uca_ema_decay = float(np.clip(uca_ema_decay, 0.0, 0.999))
        self.last_update_stats = {}

        # get_obs = [self_id(N), nbr_id(N), pos_flat(2N), U_local(M), patch(9)]
        self.nbr_feat_start = self.team_size
        self.nbr_feat_end = 2 * self.team_size
        self.u_feat_start = 4 * self.team_size
        self.u_feat_dim = self.state_dim - self.u_feat_start - 9
        self.u_feat_end = self.u_feat_start + max(self.u_feat_dim, 0)

    def entropy_coef(self, episode):
        return max(self.beta0 * (self.beta_decay ** (episode - 1)), self.beta_min)

    def _lr_scale(self, step):
        step = max(int(step), 1)

        warmup = self.scheduler_warmup_steps
        decay_steps = self.scheduler_decay_steps
        final_rate = self.scheduler_decay_rate

        # 1) warmup: 线性从 0 -> 1
        if warmup > 0 and step <= warmup:
            return float(step) / float(warmup)

        # 2) decay: 线性从 1 -> final_rate
        if decay_steps > 0:
            # warmup 结束后的第一步仍保持 base lr，再开始线性衰减
            decay_t = max(step - warmup - 1, 0)
            progress = np.clip(float(decay_t) / float(decay_steps), 0.0, 1.0)
            return float(1.0 - (1.0 - final_rate) * progress)

        # 3) no decay
        return 1.0

    def _apply_lr_schedule(self, step):
        scale = self._lr_scale(step)
        actor_lr = self.actor_base_lr * scale
        critic_lr = self.critic_base_lr * scale
        for g in self.actor_optimizer.param_groups:
            g["lr"] = actor_lr
        for g in self.critic_optimizer.param_groups:
            g["lr"] = critic_lr
        return float(actor_lr), float(critic_lr)

    def save_model(self, results_dir, subdir=None):
        if subdir is None:
            subdir = self.weights_subdir
        path = os.path.join(results_dir, subdir)
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load_model(self, path=None):
        if path is None:
            path = self.weights_subdir
        actor_path = os.path.join(path, "actor.pth")
        critic_path = os.path.join(path, "critic.pth")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

    def _optimizer_to_device(self, optimizer):
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(self.device)

    def get_resume_state(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "last_update_stats": dict(self.last_update_stats),
        }

    def load_resume_state(self, state):
        if not isinstance(state, dict):
            return
        actor_opt = state.get("actor_optimizer")
        critic_opt = state.get("critic_optimizer")
        if actor_opt is not None:
            self.actor_optimizer.load_state_dict(actor_opt)
            self._optimizer_to_device(self.actor_optimizer)
        if critic_opt is not None:
            self.critic_optimizer.load_state_dict(critic_opt)
            self._optimizer_to_device(self.critic_optimizer)
        last_stats = state.get("last_update_stats")
        if isinstance(last_stats, dict):
            self.last_update_stats = dict(last_stats)

    def take_action(self, state_all_agent):
        s = torch.as_tensor(np.asarray(state_all_agent, dtype=np.float32), device=self.device)
        probs = self.actor(s)

        env = env_core.CURRENT_ENV
        if env is None:
            raise RuntimeError("No active EnvCore instance registered (env_core.CURRENT_ENV is None).")
        masks = torch.as_tensor(np.asarray(env.get_action_masks(), dtype=np.float32), device=self.device)

        masked = probs * masks
        row_sums = masked.sum(dim=1, keepdim=True)
        zero_rows = row_sums <= 0.0
        if zero_rows.any():
            allowed_counts = masks.sum(dim=1, keepdim=True).clamp(min=1.0)
            masked = torch.where(allowed_counts > 0, masks / allowed_counts, masks)
        else:
            masked = masked / row_sums

        action_dist = torch.distributions.Categorical(masked)
        actions_t = action_dist.sample()
        actions = actions_t.detach().cpu().tolist()
        probs_np = masked.detach().cpu().numpy()
        action_probs = [probs_np[i] for i in range(probs_np.shape[0])]
        return actions, action_probs

    def _uniform_weights(self, T):
        return np.full((self.team_size, T), 1.0 / float(self.team_size), dtype=np.float32)

    def _smooth_weights(self, w):
        # w shape: [N, T]
        # rho=0 -> 完全均匀分配；rho=1 -> 完全使用原始贡献权重
        uniform = np.full_like(w, 1.0 / float(self.team_size), dtype=np.float32)
        return ((1.0 - self.uca_rho) * uniform + self.uca_rho * w).astype(np.float32)

    def _ema_smooth_contrib(self, contrib):
        if contrib is None or contrib.size == 0:
            return contrib
        if self.uca_ema_decay <= 0.0 or contrib.shape[1] <= 1:
            return contrib.astype(np.float32)

        smoothed = np.empty_like(contrib, dtype=np.float32)
        smoothed[:, 0] = contrib[:, 0]
        keep = self.uca_ema_decay
        add = 1.0 - keep
        for t in range(1, contrib.shape[1]):
            smoothed[:, t] = keep * smoothed[:, t - 1] + add * contrib[:, t]
        return smoothed

    def _compute_contrib(self, transition_dicts, T):
        if self.u_feat_dim <= 0:
            return None, None

        self_du = np.zeros((self.team_size, T), dtype=np.float32)
        nbr_mask = np.zeros((self.team_size, T, self.team_size), dtype=np.float32)
        for i in range(self.team_size):
            s_i = np.asarray(transition_dicts[i]["states"], dtype=np.float32)
            ns_i = np.asarray(transition_dicts[i]["next_states"], dtype=np.float32)
            if s_i.ndim != 2 or ns_i.ndim != 2:
                return None, None
            if s_i.shape[1] < self.u_feat_end or ns_i.shape[1] < self.u_feat_end:
                return None, None
            if s_i.shape[1] < self.nbr_feat_end:
                return None, None

            u_now = s_i[:, self.u_feat_start : self.u_feat_end]
            u_next = ns_i[:, self.u_feat_start : self.u_feat_end]
            du = np.maximum(u_now - u_next, 0.0)
            self_du[i] = du.sum(axis=1)
            nbr_mask[i] = (s_i[:, self.nbr_feat_start : self.nbr_feat_end] > 0.0).astype(np.float32)
            nbr_mask[i, :, i] = 1.0

        joint_contrib = np.einsum("itj,jt->it", nbr_mask, self_du, dtype=np.float32)
        joint_contrib = np.maximum(joint_contrib, 0.0).astype(np.float32)
        smoothed_contrib = self._ema_smooth_contrib(joint_contrib)
        return smoothed_contrib, joint_contrib

    def _uncertainty_weights(self, contrib, T):
        if contrib is None:
            return self._uniform_weights(T)

        if self.uca_weight_norm == "linear":
            denom = np.sum(contrib, axis=0, keepdims=True)
            weights = contrib / np.clip(denom, 1e-8, None)
        else:
            x = contrib / self.uca_tau
            x = x - np.max(x, axis=0, keepdims=True)
            exp_x = np.exp(x)
            denom = np.sum(exp_x, axis=0, keepdims=True)
            weights = exp_x / np.clip(denom, 1e-8, None)

        bad_cols = np.where((denom <= 0.0).reshape(-1))[0]
        if bad_cols.size > 0:
            weights[:, bad_cols] = 1.0 / float(self.team_size)

        if not np.isfinite(weights).all():
            return self._uniform_weights(T)
        return weights.astype(np.float32)

    def _lambda_t(self, contrib, T):
        # 双门控: lambda_t = theta * g_m * g_p
        m_gate_disabled = self.uca_m_min <= 0.0
        p_gate_disabled = self.uca_p_min <= 0.0

        # 两个门都关掉 -> 不做动态门控，仅保留固定幅度 theta
        if m_gate_disabled and p_gate_disabled:
            ones = np.ones((T,), dtype=np.float32)
            lam = np.full((T,), self.uca_theta, dtype=np.float32)
            zeros = np.zeros((T,), dtype=np.float32)
            return lam, ones, ones, zeros, zeros

        if contrib is None:
            zeros = np.zeros((T,), dtype=np.float32)
            g_m = np.ones((T,), dtype=np.float32) if m_gate_disabled else zeros
            g_p = np.ones((T,), dtype=np.float32) if p_gate_disabled else zeros
            lam = self.uca_theta * g_m * g_p
            return lam.astype(np.float32), g_m.astype(np.float32), g_p.astype(np.float32), zeros, zeros

        m_t = contrib.mean(axis=0)  # 每步平均贡献幅值
        # 有效参与度(均匀性门控):
        # neff = 1/sum(c_hat^2), c_hat=c/sum(c)
        # p_eff = (neff-1)/(N-1), N>1, 取值[0,1]
        c_pos = np.maximum(contrib, 0.0).astype(np.float32)
        sum_c = c_pos.sum(axis=0)  # [T]
        p_eff = np.zeros((T,), dtype=np.float32)
        valid = sum_c > 1e-8
        if np.any(valid):
            c_hat = c_pos[:, valid] / sum_c[valid][None, :]
            h = np.sum(c_hat * c_hat, axis=0)
            neff = 1.0 / np.clip(h, 1e-8, None)
            if self.team_size > 1:
                p_eff[valid] = (neff - 1.0) / float(self.team_size - 1)
            else:
                p_eff[valid] = 1.0
        p_eff = np.clip(p_eff, 0.0, 1.0)

        if m_gate_disabled:
            g_m = np.ones((T,), dtype=np.float32)
        else:
            g_m = np.clip(m_t / max(self.uca_m_min, 1e-8), 0.0, 1.0)

        if p_gate_disabled:
            g_p = np.ones((T,), dtype=np.float32)
        else:
            g_p = np.clip(p_eff / np.clip(self.uca_p_min, 1e-8, None), 0.0, 1.0)

        lam = self.uca_theta * g_m * g_p
        if not np.isfinite(lam).all():
            z = np.zeros((T,), dtype=np.float32)
            return z, z, z, z, z
        return (
            lam.astype(np.float32),
            g_m.astype(np.float32),
            g_p.astype(np.float32),
            m_t.astype(np.float32),
            p_eff.astype(np.float32),
        )

    def update(self, transition_dicts, minibatch_size, epoch_num, episode):
        actor_lr_now, critic_lr_now = self._apply_lr_schedule(episode)
        beta = self.entropy_coef(episode)
        T = len(transition_dicts[0]["states"])

        states_global = []
        next_states_global = []
        for t in range(T):
            concat_state = []
            concat_next_state = []
            for i in range(self.team_size):
                concat_state.append(transition_dicts[i]["states"][t])
                concat_next_state.append(transition_dicts[i]["next_states"][t])
            states_global.append(np.concatenate(concat_state))
            next_states_global.append(np.concatenate(concat_next_state))

        states_global = torch.as_tensor(np.asarray(states_global, dtype=np.float32), device=self.device)
        next_states_global = torch.as_tensor(np.asarray(next_states_global, dtype=np.float32), device=self.device)

        rewards_all = torch.as_tensor(
            np.asarray(
                [[transition_dicts[i]["rewards"][t] for i in range(self.team_size)] for t in range(T)],
                dtype=np.float32,
            ),
            device=self.device,
        )
        dones_all = torch.as_tensor(
            np.asarray(
                [[transition_dicts[i]["dones"][t] for i in range(self.team_size)] for t in range(T)],
                dtype=np.float32,
            ),
            device=self.device,
        )

        values_global = self.critic(states_global).squeeze(-1)
        next_values_global = self.critic(next_states_global).squeeze(-1)
        reward_global = rewards_all.mean(dim=1)
        done = dones_all.max(dim=1).values
        td_target = reward_global + self.gamma * next_values_global * (1.0 - done)
        td_delta = td_target - values_global
        adv_global = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)  # [T]

        # UCA 核心：用不确定度贡献权重重加权每个 UAV 的优势
        # phi_i,t = (N * w_i,t - 1) / (N - 1), lambda_t = g_m * g_p * theta
        # scale_i,t = 1 + lambda_t * phi_i,t
        # A_i,t = scale_i,t * A_global,t
        contrib_np, contrib_raw_np = self._compute_contrib(transition_dicts, T)
        w_np = self._uncertainty_weights(contrib_np, T)  # [N,T]
        w_np = self._smooth_weights(w_np)  # 权重平滑
        lambda_t_np, g_m_np, g_p_np, m_t_np, p_eff_np = self._lambda_t(contrib_np, T)  # [T]
        self.last_update_stats = {
            # 继续复用 alpha_t_* 这组键，便于兼容现有日志/保存流程；其数值现在对应 lambda_t
            "alpha_t_mean": float(np.mean(lambda_t_np)),
            "alpha_t_std": float(np.std(lambda_t_np)),
            "alpha_t_min": float(np.min(lambda_t_np)),
            "alpha_t_max": float(np.max(lambda_t_np)),
            "g_m_mean": float(np.mean(g_m_np)),
            "g_p_mean": float(np.mean(g_p_np)),
            "m_t_mean": float(np.mean(m_t_np)),
            "p_eff_mean": float(np.mean(p_eff_np)),
            "contrib_raw_mean": float(np.mean(contrib_raw_np)) if contrib_raw_np is not None else 0.0,
            "contrib_ema_mean": float(np.mean(contrib_np)) if contrib_np is not None else 0.0,
            "actor_lr": actor_lr_now,
            "critic_lr": critic_lr_now,
        }
        w = torch.as_tensor(w_np, dtype=torch.float32, device=self.device)
        lambda_t = torch.as_tensor(lambda_t_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.team_size > 1:
            phi = (self.team_size * w - 1.0) / float(self.team_size - 1)
        else:
            phi = torch.zeros_like(w)
        scale = 1.0 + lambda_t * phi
        adv_all = scale * adv_global.unsqueeze(0).expand(self.team_size, -1).contiguous()

        with torch.no_grad():
            mean = adv_all.mean()
            std = adv_all.std(unbiased=False).clamp_min(1e-8)
            adv_all_norm = (adv_all - mean) / std

        critic_losses = []
        action_losses = []
        entropies = []
        kls = []
        clip_fracs = []

        for _ in range(epoch_num):
            idx_all = np.random.permutation(T)
            mb_size = T if (minibatch_size is None or minibatch_size <= 0) else min(minibatch_size, T)
            num_full = T // mb_size
            if num_full == 0:
                num_full = 1
                mb_size = T
            idx_full = idx_all[: num_full * mb_size].reshape(num_full, mb_size)

            for mb_idx_np in idx_full:
                mb_idx = torch.as_tensor(mb_idx_np, device=self.device)
                values_mb = self.critic(states_global[mb_idx]).squeeze(-1)
                td_target_mb = td_target[mb_idx].detach()
                critic_loss = F.mse_loss(values_mb, td_target_mb)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                critic_losses.append(critic_loss.item())

            EPS = 1e-8
            epoch_kls = []
            early_stop_epoch = False
            orig_lrs = [g["lr"] for g in self.actor_optimizer.param_groups]

            for mb_idx_np in idx_full:
                mb_idx = torch.as_tensor(mb_idx_np, device=self.device)
                total_actor_loss = 0.0
                batch_action_losses = []
                batch_entropies = []
                batch_kls = []
                batch_frac_clipped = []

                for i in range(self.team_size):
                    states = torch.as_tensor(
                        np.asarray(transition_dicts[i]["states"], dtype=np.float32), device=self.device
                    )
                    actions = torch.as_tensor(transition_dicts[i]["actions"], device=self.device).view(-1, 1)
                    old_probs = torch.as_tensor(
                        np.asarray(transition_dicts[i]["action_probs"], dtype=np.float32), device=self.device
                    )

                    states_mb = states[mb_idx]
                    actions_mb = actions[mb_idx]
                    old_probs_mb = old_probs[mb_idx]
                    adv_mb = adv_all_norm[i, mb_idx].unsqueeze(-1)

                    mask_mb = (old_probs_mb > 0).to(old_probs_mb.dtype)
                    current_probs_raw = self.actor(states_mb)
                    current_probs = current_probs_raw * mask_mb
                    row_sums = current_probs.sum(dim=1, keepdim=True)
                    zero_rows = row_sums <= 0
                    if zero_rows.any():
                        allowed_counts = mask_mb.sum(dim=1, keepdim=True).clamp(min=1.0)
                        safe_uniform = mask_mb / allowed_counts
                        current_probs = torch.where(zero_rows, safe_uniform, current_probs)
                        row_sums = current_probs.sum(dim=1, keepdim=True)
                    current_probs = current_probs / row_sums.clamp_min(EPS)

                    with torch.no_grad():
                        kl_per_agent = (
                            old_probs_mb.clamp_min(EPS)
                            * (
                                torch.log(old_probs_mb.clamp_min(EPS))
                                - torch.log(current_probs.clamp_min(EPS))
                            )
                        ).sum(dim=1).mean()
                        batch_kls.append(kl_per_agent.item())

                    cur_sel = current_probs.gather(1, actions_mb).clamp_min(EPS)
                    old_sel = old_probs_mb.gather(1, actions_mb).clamp_min(EPS)
                    cur_log_probs = torch.log(cur_sel)
                    old_log_probs = torch.log(old_sel).detach()
                    ratio = torch.exp(cur_log_probs - old_log_probs)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_mb

                    frac_clip_i = ((ratio > 1.0 + self.eps) | (ratio < 1.0 - self.eps)).float().mean().item()
                    batch_frac_clipped.append(frac_clip_i)

                    action_loss = torch.mean(-torch.min(surr1, surr2))
                    entropy_val = compute_entropy(current_probs)
                    entropy_t = torch.distributions.Categorical(probs=current_probs).entropy().mean()

                    total_actor_loss = total_actor_loss + (action_loss - beta * entropy_t)
                    batch_action_losses.append(action_loss.item())
                    batch_entropies.append(entropy_val)

                action_losses.append(float(np.mean(batch_action_losses)))
                entropies.append(float(np.mean(batch_entropies)))
                mb_kl = float(np.mean(batch_kls))
                clip_fracs.append(float(np.mean(batch_frac_clipped)))
                epoch_kls.append(mb_kl)
                kls.append(mb_kl)

                if mb_kl > 2.0 * self.target_kl:
                    early_stop_epoch = True
                    break
                elif mb_kl > 1.5 * self.target_kl:
                    for g in self.actor_optimizer.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, 1e-6)

                total_actor_loss = total_actor_loss / self.team_size
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                self.actor_optimizer.step()

            for g, lr in zip(self.actor_optimizer.param_groups, orig_lrs):
                g["lr"] = lr

            if epoch_kls:
                epoch_avg_kl = float(np.mean(epoch_kls))
                if epoch_avg_kl > 1.5 * self.target_kl:
                    early_stop_epoch = True

            if early_stop_epoch:
                break

        return (
            float(np.mean(action_losses)),
            float(np.mean(critic_losses)),
            float(np.mean(entropies)),
            float(np.mean(kls)),
            float(np.mean(clip_fracs)),
            float(np.mean(lambda_t_np)),
            float(np.std(lambda_t_np)),
            float(np.min(lambda_t_np)),
            float(np.max(lambda_t_np)),
        )
