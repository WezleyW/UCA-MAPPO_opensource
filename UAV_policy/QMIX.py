import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import env_core


class AgentQNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QMixer(torch.nn.Module):
    """
    QMIX mixing network:
      Q_tot = f(state, q_1, ..., q_n), and f is monotonic in each q_i.
    """

    def __init__(self, n_agents, state_dim, mixing_embed_dim, hypernet_embed_dim):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.mixing_embed_dim = int(mixing_embed_dim)

        self.hyper_w1 = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, int(hypernet_embed_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hypernet_embed_dim), self.n_agents * self.mixing_embed_dim),
        )
        self.hyper_b1 = torch.nn.Linear(self.state_dim, self.mixing_embed_dim)

        self.hyper_w_final = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, int(hypernet_embed_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hypernet_embed_dim), self.mixing_embed_dim),
        )

        self.v_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.mixing_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mixing_embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        # agent_qs: [B, N], states: [B, state_dim]
        B = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states)).view(B, self.n_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.mixing_embed_dim)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # [B,1,E]

        w_final = torch.abs(self.hyper_w_final(states)).view(B, self.mixing_embed_dim, 1)
        v = self.v_net(states).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w_final) + v  # [B,1,1]
        return q_tot.view(B, 1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.obs = deque(maxlen=self.capacity)               # [N,S]
        self.actions = deque(maxlen=self.capacity)           # [N]
        self.rewards = deque(maxlen=self.capacity)           # scalar
        self.next_obs = deque(maxlen=self.capacity)          # [N,S]
        self.dones = deque(maxlen=self.capacity)             # scalar
        self.next_action_masks = deque(maxlen=self.capacity) # [N,A]

    def __len__(self):
        return len(self.rewards)

    def add(self, obs, actions, rewards, next_obs, done, next_action_masks):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(float(rewards))
        self.next_obs.append(next_obs)
        self.dones.append(float(done))
        self.next_action_masks.append(next_action_masks)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.rewards), size=int(batch_size))
        return {
            "obs": np.asarray([self.obs[i] for i in idx], dtype=np.float32),  # [B,N,S]
            "actions": np.asarray([self.actions[i] for i in idx], dtype=np.int64),  # [B,N]
            "rewards": np.asarray([self.rewards[i] for i in idx], dtype=np.float32),  # [B]
            "next_obs": np.asarray([self.next_obs[i] for i in idx], dtype=np.float32),  # [B,N,S]
            "dones": np.asarray([self.dones[i] for i in idx], dtype=np.float32),  # [B]
            "next_action_masks": np.asarray([self.next_action_masks[i] for i in idx], dtype=np.float32),  # [B,N,A]
        }

    def state_dict(self):
        return {
            "capacity": int(self.capacity),
            "obs": list(self.obs),
            "actions": list(self.actions),
            "rewards": list(self.rewards),
            "next_obs": list(self.next_obs),
            "dones": list(self.dones),
            "next_action_masks": list(self.next_action_masks),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        capacity = int(state.get("capacity", self.capacity))
        self.capacity = capacity
        self.obs = deque(state.get("obs", []), maxlen=capacity)
        self.actions = deque(state.get("actions", []), maxlen=capacity)
        self.rewards = deque(state.get("rewards", []), maxlen=capacity)
        self.next_obs = deque(state.get("next_obs", []), maxlen=capacity)
        self.dones = deque(state.get("dones", []), maxlen=capacity)
        self.next_action_masks = deque(state.get("next_action_masks", []), maxlen=capacity)


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
        agent_hidden_dim = int(policy_cfg["agent_hidden_dim"])
        mixing_embed_dim = int(policy_cfg["mixing_embed_dim"])
        hypernet_embed_dim = int(policy_cfg["hypernet_embed_dim"])
        q_lr = float(policy_cfg["q_lr"])
        scheduler_warmup_steps = int(policy_cfg.get("scheduler_warmup_steps", 0))
        scheduler_decay_steps = int(policy_cfg.get("scheduler_decay_steps", 0))
        scheduler_decay_rate = float(policy_cfg.get("scheduler_decay_rate", 1.0))
        gamma = float(policy_cfg["gamma"])
        epsilon_start = float(policy_cfg["epsilon_start"])
        epsilon_end = float(policy_cfg["epsilon_end"])
        epsilon_decay = float(policy_cfg["epsilon_decay"])
        replay_size = int(policy_cfg["replay_size"])
        min_replay_size = int(policy_cfg["min_replay_size"])
        batch_size = int(policy_cfg["batch_size"])
        target_update_interval = int(policy_cfg["target_update_interval"])
        method_name = str(method)

        self.n_agents = int(team_size)
        self.obs_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.state_dim = self.n_agents * self.obs_dim
        self.device = device

        self.gamma = float(gamma)
        self.q_base_lr = float(q_lr)
        self.scheduler_warmup_steps = max(int(scheduler_warmup_steps), 0)
        self.scheduler_decay_steps = max(int(scheduler_decay_steps), 0)
        self.scheduler_decay_rate = float(np.clip(scheduler_decay_rate, 0.0, 1.0))
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = max(float(epsilon_decay), 1.0)
        self.min_replay_size = int(min_replay_size)
        self.batch_size = int(batch_size)
        self.target_update_interval = max(int(target_update_interval), 1)

        self.method_name = method_name
        self.weights_subdir = f"{self.method_name}_weights"

        self.agent_q = AgentQNet(self.obs_dim, int(agent_hidden_dim), self.action_dim).to(self.device)
        self.target_agent_q = AgentQNet(self.obs_dim, int(agent_hidden_dim), self.action_dim).to(self.device)
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())

        self.mixer = QMixer(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            mixing_embed_dim=int(mixing_embed_dim),
            hypernet_embed_dim=int(hypernet_embed_dim),
        ).to(self.device)
        self.target_mixer = QMixer(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            mixing_embed_dim=int(mixing_embed_dim),
            hypernet_embed_dim=int(hypernet_embed_dim),
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()),
            lr=self.q_base_lr,
        )
        self.replay = ReplayBuffer(replay_size)

        self.total_env_steps = 0
        self.total_update_steps = 0

        # QMIX 主要统计回报、时长、Q损失
        self.plot_metric_keys = [
            "Total_Reward_per_Episode",
            "Episode_Length_per_Episode",
            "Value_Loss_per_Episode",
        ]
        self.extra_log_metrics = ["value_loss"]
        self.extra_save_metrics = ["value_loss"]

    def _epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -float(self.total_env_steps) / self.epsilon_decay
        )

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
        q_lr = self.q_base_lr * self._lr_scale(step)
        for g in self.optimizer.param_groups:
            g["lr"] = q_lr
        return float(q_lr)

    def save_model(self, results_dir, subdir=None):
        if subdir is None:
            subdir = self.weights_subdir
        path = os.path.join(results_dir, subdir)
        os.makedirs(path, exist_ok=True)
        torch.save(self.agent_q.state_dict(), os.path.join(path, "agent_q.pth"))
        torch.save(self.mixer.state_dict(), os.path.join(path, "mixer.pth"))
        torch.save(self.target_agent_q.state_dict(), os.path.join(path, "target_agent_q.pth"))
        torch.save(self.target_mixer.state_dict(), os.path.join(path, "target_mixer.pth"))

    def load_model(self, path=None):
        if path is None:
            path = self.weights_subdir
        agent_q_path = os.path.join(path, "agent_q.pth")
        mixer_path = os.path.join(path, "mixer.pth")
        target_agent_q_path = os.path.join(path, "target_agent_q.pth")
        target_mixer_path = os.path.join(path, "target_mixer.pth")
        if os.path.exists(agent_q_path):
            self.agent_q.load_state_dict(torch.load(agent_q_path, map_location=self.device))
        if os.path.exists(mixer_path):
            self.mixer.load_state_dict(torch.load(mixer_path, map_location=self.device))
        if os.path.exists(target_agent_q_path):
            self.target_agent_q.load_state_dict(torch.load(target_agent_q_path, map_location=self.device))
        if os.path.exists(target_mixer_path):
            self.target_mixer.load_state_dict(torch.load(target_mixer_path, map_location=self.device))

    def _optimizer_to_device(self, optimizer):
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(self.device)

    def get_resume_state(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "total_env_steps": int(self.total_env_steps),
            "total_update_steps": int(self.total_update_steps),
        }

    def load_resume_state(self, state):
        if not isinstance(state, dict):
            return
        opt_state = state.get("optimizer")
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
            self._optimizer_to_device(self.optimizer)
        replay_state = state.get("replay")
        if replay_state is not None:
            self.replay.load_state_dict(replay_state)
        if "total_env_steps" in state:
            self.total_env_steps = int(state["total_env_steps"])
        if "total_update_steps" in state:
            self.total_update_steps = int(state["total_update_steps"])

    def take_action(self, state_all_agent):
        obs = np.asarray(state_all_agent, dtype=np.float32)  # [N,S]
        obs_t = torch.as_tensor(obs, device=self.device)
        with torch.no_grad():
            q_vals = self.agent_q(obs_t).detach().cpu().numpy()  # [N,A]

        env = env_core.CURRENT_ENV
        if env is None:
            raise RuntimeError("No active EnvCore instance registered (env_core.CURRENT_ENV is None).")
        masks = np.asarray(env.get_action_masks(), dtype=np.float32)  # [N,A]

        eps = self._epsilon()
        actions = []
        for i in range(self.n_agents):
            legal = np.flatnonzero(masks[i] > 0.0)
            if legal.size == 0:
                raise RuntimeError(f"No legal actions for agent {i}.")
            if np.random.rand() < eps:
                action = int(np.random.choice(legal))
            else:
                action = int(legal[int(np.argmax(q_vals[i, legal]))])
            actions.append(action)

        self.total_env_steps += 1
        # 为兼容 train_policy 的 buffer 字段，返回动作掩码
        return actions, [masks[i].copy() for i in range(self.n_agents)]

    def _append_rollout_to_replay(self, transition_dicts):
        T = len(transition_dicts[0]["states"])
        for t in range(T):
            obs_t = np.asarray(
                [transition_dicts[i]["states"][t] for i in range(self.n_agents)],
                dtype=np.float32,
            )  # [N,S]
            actions_t = np.asarray(
                [transition_dicts[i]["actions"][t] for i in range(self.n_agents)],
                dtype=np.int64,
            )  # [N]
            next_obs_t = np.asarray(
                [transition_dicts[i]["next_states"][t] for i in range(self.n_agents)],
                dtype=np.float32,
            )  # [N,S]
            next_masks_t = np.asarray(
                [transition_dicts[i]["next_action_masks"][t] for i in range(self.n_agents)],
                dtype=np.float32,
            )  # [N,A]
            # 环境当前是全局奖励广播到每个 UAV，这里取均值做团队奖励更稳妥
            reward_t = float(
                np.mean([transition_dicts[i]["rewards"][t] for i in range(self.n_agents)], dtype=np.float32)
            )
            done_t = float(
                np.max([transition_dicts[i]["dones"][t] for i in range(self.n_agents)], initial=0.0)
            )
            self.replay.add(obs_t, actions_t, reward_t, next_obs_t, done_t, next_masks_t)

    def update(self, transition_dicts, minibatch_size, epoch_num, episode):
        self._apply_lr_schedule(episode)
        self._append_rollout_to_replay(transition_dicts)

        if len(self.replay) < max(self.min_replay_size, 2):
            # 与 train_policy 返回对齐：
            # (policy_loss, value_loss, entropy, kl, clip_frac,
            #  alpha_t_mean, alpha_t_std, alpha_t_min, alpha_t_max)
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if minibatch_size is None or minibatch_size <= 0:
            batch_size = min(self.batch_size, len(self.replay))
        else:
            batch_size = min(int(minibatch_size), len(self.replay))
        updates = max(1, int(epoch_num))

        losses = []
        for _ in range(updates):
            batch = self.replay.sample(batch_size)
            obs = torch.as_tensor(batch["obs"], device=self.device)  # [B,N,S]
            actions = torch.as_tensor(batch["actions"], device=self.device).long()  # [B,N]
            rewards = torch.as_tensor(batch["rewards"], device=self.device).unsqueeze(-1)  # [B,1]
            next_obs = torch.as_tensor(batch["next_obs"], device=self.device)  # [B,N,S]
            dones = torch.as_tensor(batch["dones"], device=self.device).unsqueeze(-1)  # [B,1]
            next_masks = torch.as_tensor(batch["next_action_masks"], device=self.device)  # [B,N,A]

            B = obs.shape[0]
            obs_flat = obs.reshape(B * self.n_agents, self.obs_dim)
            next_obs_flat = next_obs.reshape(B * self.n_agents, self.obs_dim)

            # 在线网络估计当前 Q_i(o_i, a_i)
            q_all = self.agent_q(obs_flat).reshape(B, self.n_agents, self.action_dim)
            chosen_q = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B,N]

            state = obs.reshape(B, self.state_dim)
            q_tot = self.mixer(chosen_q, state)  # [B,1]

            with torch.no_grad():
                # Double Q: online 选动作，target 评估
                online_next_q_all = self.agent_q(next_obs_flat).reshape(B, self.n_agents, self.action_dim)
                masked_online_next_q = online_next_q_all.masked_fill(next_masks <= 0.0, -1e9)
                next_actions = masked_online_next_q.argmax(dim=2, keepdim=True)  # [B,N,1]

                target_next_q_all = self.target_agent_q(next_obs_flat).reshape(B, self.n_agents, self.action_dim)
                target_next_q = target_next_q_all.gather(2, next_actions).squeeze(-1)  # [B,N]

                next_state = next_obs.reshape(B, self.state_dim)
                target_q_tot = self.target_mixer(target_next_q, next_state)  # [B,1]
                y = rewards + self.gamma * (1.0 - dones) * target_q_tot

            loss = F.mse_loss(q_tot, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(float(loss.item()))
            self.total_update_steps += 1
            if self.total_update_steps % self.target_update_interval == 0:
                self.target_agent_q.load_state_dict(self.agent_q.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        # 对齐 train_policy 返回格式，把 QMix 损失放在 value_loss；
        # alpha 相关指标对 QMIX 不适用，返回 0.0 占位。
        return 0.0, mean_loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
