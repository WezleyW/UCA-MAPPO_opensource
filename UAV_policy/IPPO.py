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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _gae(gamma, lmbda, td_delta):
    # td_delta: [T]
    out = []
    adv = 0.0
    for delta in td_delta[::-1]:
        adv = gamma * lmbda * adv + delta
        out.append(adv)
    out.reverse()
    return np.asarray(out, dtype=np.float32)


class UAV_Policy:
    @classmethod
    def from_config(
        cls,
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
        gamma = float(policy_cfg["gamma"])
        lmbda = float(policy_cfg["lmbda"])
        eps = float(policy_cfg["eps"])
        target_kl = float(policy_cfg["target_kl"])
        beta0 = float(policy_cfg["beta0"])
        beta_decay = float(policy_cfg["beta_decay"])
        beta_min = float(policy_cfg["beta_min"])
        scheduler_warmup_steps = int(policy_cfg.get("scheduler_warmup_steps", 0))
        scheduler_decay_steps = int(policy_cfg.get("scheduler_decay_steps", 0))
        scheduler_decay_rate = float(policy_cfg.get("scheduler_decay_rate", 1.0))
        max_grad_norm_actor = float(policy_cfg.get("max_grad_norm_actor", 0.5))
        max_grad_norm_critic = float(policy_cfg.get("max_grad_norm_critic", 1.0))
        adv_clip = float(policy_cfg.get("adv_clip", 5.0))
        ratio_max = float(policy_cfg.get("ratio_max", 4.0))
        method_name = str(method)

        self.team_size = int(team_size)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.lmbda = float(lmbda)
        self.eps = float(eps)
        self.target_kl = float(target_kl)
        self.beta0 = float(beta0)
        self.beta_decay = float(beta_decay)
        self.beta_min = float(beta_min)
        self.device = device
        self.actor_base_lr = float(actor_lr)
        self.critic_base_lr = float(critic_lr)
        self.scheduler_warmup_steps = max(int(scheduler_warmup_steps), 0)
        self.scheduler_decay_steps = max(int(scheduler_decay_steps), 0)
        self.scheduler_decay_rate = float(np.clip(scheduler_decay_rate, 0.0, 1.0))
        self.max_grad_norm_actor = max(float(max_grad_norm_actor), 0.0)
        self.max_grad_norm_critic = max(float(max_grad_norm_critic), 0.0)
        self.adv_clip = max(float(adv_clip), 0.0)
        self.ratio_max = max(float(ratio_max), 1.0)

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
        self.extra_log_metrics = ["policy_loss", "value_loss", "entropy", "kl", "clip_frac"]
        self.extra_save_metrics = ["policy_loss", "value_loss", "entropy", "kl", "clip_frac"]

        # IPPO 常见设置：参数共享 actor + 参数共享局部 critic
        self.actor = PolicyNet(self.state_dim, int(hidden_dim), self.action_dim).to(self.device)
        self.critic = ValueNet(self.state_dim, int(hidden_dim)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_base_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_base_lr)

    def entropy_coef(self, episode):
        return max(self.beta0 * (self.beta_decay ** (episode - 1)), self.beta_min)

    def _lr_scale(self, step):
        step = max(int(step), 1)

        warmup = self.scheduler_warmup_steps
        decay_steps = self.scheduler_decay_steps
        final_rate = self.scheduler_decay_rate

        if warmup > 0 and step <= warmup:
            return float(step) / float(warmup)

        if decay_steps > 0:
            decay_t = max(step - warmup - 1, 0)
            progress = np.clip(float(decay_t) / float(decay_steps), 0.0, 1.0)
            return float(1.0 - (1.0 - final_rate) * progress)

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

    def take_action(self, state_all_agent):
        s = torch.as_tensor(np.asarray(state_all_agent, dtype=np.float32), device=self.device)
        probs = self.actor(s)  # [N, A]

        env = env_core.CURRENT_ENV
        if env is None:
            raise RuntimeError("No active EnvCore instance registered (env_core.CURRENT_ENV is None).")
        masks = torch.as_tensor(np.asarray(env.get_action_masks(), dtype=np.float32), device=self.device)

        masked = probs * masks
        row_sums = masked.sum(dim=1, keepdim=True)
        zero_rows = row_sums <= 0.0
        if zero_rows.any():
            allowed = masks.sum(dim=1, keepdim=True).clamp(min=1.0)
            safe_uniform = masks / allowed
            masked = torch.where(zero_rows, safe_uniform, masked)
            row_sums = masked.sum(dim=1, keepdim=True)
        masked = masked / row_sums.clamp_min(1e-8)

        action_dist = torch.distributions.Categorical(masked)
        actions_t = action_dist.sample()
        actions = actions_t.detach().cpu().tolist()
        probs_np = masked.detach().cpu().numpy()
        action_probs = [probs_np[i] for i in range(probs_np.shape[0])]
        return actions, action_probs

    def update(self, transition_dicts, minibatch_size, epoch_num, episode):
        self._apply_lr_schedule(episode)
        beta = self.entropy_coef(episode)
        T = len(transition_dicts[0]["states"])
        N = self.team_size
        EPS = 1e-8

        # 1) 组装 [N, T, *] 数据
        states = np.asarray(
            [[transition_dicts[i]["states"][t] for t in range(T)] for i in range(N)],
            dtype=np.float32,
        )  # [N, T, S]
        next_states = np.asarray(
            [[transition_dicts[i]["next_states"][t] for t in range(T)] for i in range(N)],
            dtype=np.float32,
        )  # [N, T, S]
        rewards = np.asarray(
            [[transition_dicts[i]["rewards"][t] for t in range(T)] for i in range(N)],
            dtype=np.float32,
        )  # [N, T]
        dones = np.asarray(
            [[transition_dicts[i]["dones"][t] for t in range(T)] for i in range(N)],
            dtype=np.float32,
        )  # [N, T]
        actions = np.asarray(
            [[transition_dicts[i]["actions"][t] for t in range(T)] for i in range(N)],
            dtype=np.int64,
        )  # [N, T]
        old_probs = np.asarray(
            [[transition_dicts[i]["action_probs"][t] for t in range(T)] for i in range(N)],
            dtype=np.float32,
        )  # [N, T, A]

        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        # 2) 局部 critic 计算 GAE：每个 UAV 各自 V(o_i)
        with torch.no_grad():
            v = self.critic(states_t.reshape(-1, self.state_dim)).reshape(N, T)  # [N,T]
            v_next = self.critic(next_states_t.reshape(-1, self.state_dim)).reshape(N, T)  # [N,T]
            td_target = rewards_t + self.gamma * (1.0 - dones_t) * v_next
            td_delta = (td_target - v).detach().cpu().numpy()

        adv = np.zeros((N, T), dtype=np.float32)
        for i in range(N):
            adv[i] = _gae(self.gamma, self.lmbda, td_delta[i])
        adv_t = torch.as_tensor(adv, device=self.device)
        with torch.no_grad():
            # 局部 critic 的尺度漂移会比 centralized critic 更明显，逐 agent 归一化更稳。
            adv_mean = adv_t.mean(dim=1, keepdim=True)
            adv_std = adv_t.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
            adv_t = (adv_t - adv_mean) / adv_std
            if self.adv_clip > 0.0:
                adv_t = torch.clamp(adv_t, -self.adv_clip, self.adv_clip)

        # 3) 扁平化样本维：M = N*T
        M = N * T
        states_flat = torch.as_tensor(states.reshape(M, self.state_dim), device=self.device)
        actions_flat = torch.as_tensor(actions.reshape(M), device=self.device).long().unsqueeze(-1)
        old_probs_flat = torch.as_tensor(old_probs.reshape(M, self.action_dim), device=self.device)
        adv_flat = adv_t.reshape(M, 1)
        td_target_flat = td_target.reshape(M, 1).detach()

        mb_size = M if (minibatch_size is None or minibatch_size <= 0) else min(int(minibatch_size), M)

        action_losses = []
        critic_losses = []
        entropies = []
        kls = []
        clip_fracs = []

        for _ in range(int(epoch_num)):
            idx_all = np.random.permutation(M)
            num_full = M // mb_size
            if num_full == 0:
                num_full = 1
                mb_size = M
            idx_full = idx_all[: num_full * mb_size].reshape(num_full, mb_size)

            epoch_kls = []
            early_stop = False
            orig_lrs = [g["lr"] for g in self.actor_optimizer.param_groups]

            for mb_idx_np in idx_full:
                mb_idx = torch.as_tensor(mb_idx_np, device=self.device)
                states_mb = states_flat[mb_idx]
                actions_mb = actions_flat[mb_idx]
                old_probs_mb = old_probs_flat[mb_idx]
                adv_mb = adv_flat[mb_idx]
                td_target_mb = td_target_flat[mb_idx]

                # critic update
                v_pred_mb = self.critic(states_mb)
                critic_loss = F.mse_loss(v_pred_mb, td_target_mb)
                if not torch.isfinite(critic_loss):
                    early_stop = True
                    break
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.max_grad_norm_critic > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.item()))

                # actor update with mask reconstructed from old_probs
                mask_mb = (old_probs_mb > 0).to(old_probs_mb.dtype)
                current_raw = self.actor(states_mb)
                current = current_raw * mask_mb
                row_sums = current.sum(dim=1, keepdim=True)
                zero_rows = row_sums <= 0.0
                if zero_rows.any():
                    allowed = mask_mb.sum(dim=1, keepdim=True).clamp(min=1.0)
                    safe_uniform = mask_mb / allowed
                    current = torch.where(zero_rows, safe_uniform, current)
                    row_sums = current.sum(dim=1, keepdim=True)
                current = current / row_sums.clamp_min(EPS)

                with torch.no_grad():
                    kl_mb = (
                        old_probs_mb.clamp_min(EPS)
                        * (torch.log(old_probs_mb.clamp_min(EPS)) - torch.log(current.clamp_min(EPS)))
                    ).sum(dim=1).mean()
                    kl_val = float(kl_mb.item())
                    kls.append(kl_val)
                    epoch_kls.append(kl_val)

                cur_sel = current.gather(1, actions_mb).clamp_min(EPS)
                old_sel = old_probs_mb.gather(1, actions_mb).clamp_min(EPS)
                log_ratio = torch.log(cur_sel) - torch.log(old_sel).detach()
                ratio = torch.exp(log_ratio).clamp(min=1.0 / self.ratio_max, max=self.ratio_max)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy_t = torch.distributions.Categorical(probs=current).entropy().mean()
                entropy_val = float(entropy_t.item())
                frac_clip = float(((ratio > 1.0 + self.eps) | (ratio < 1.0 - self.eps)).float().mean().item())

                total_actor_loss = policy_loss - beta * entropy_t
                if not torch.isfinite(total_actor_loss):
                    early_stop = True
                    break
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                if self.max_grad_norm_actor > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
                self.actor_optimizer.step()

                action_losses.append(float(policy_loss.item()))
                entropies.append(entropy_val)
                clip_fracs.append(frac_clip)

                if kl_val > 2.0 * self.target_kl:
                    early_stop = True
                    break
                if kl_val > 1.5 * self.target_kl:
                    for g in self.actor_optimizer.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, 1e-6)

            for g, lr in zip(self.actor_optimizer.param_groups, orig_lrs):
                g["lr"] = lr

            if epoch_kls and float(np.mean(epoch_kls)) > 1.5 * self.target_kl:
                early_stop = True
            if early_stop:
                break

        return (
            float(np.mean(action_losses)) if action_losses else 0.0,
            float(np.mean(critic_losses)) if critic_losses else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
            float(np.mean(kls)) if kls else 0.0,
            float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
