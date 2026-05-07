import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import env_core


class QNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.states = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.next_states = deque(maxlen=self.capacity)
        self.dones = deque(maxlen=self.capacity)
        self.next_action_masks = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.actions)

    def add(self, state, action, reward, next_state, done, next_action_mask):
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(next_state)
        self.dones.append(float(done))
        self.next_action_masks.append(next_action_mask)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.actions), size=int(batch_size))
        return {
            "states": np.asarray([self.states[i] for i in idx], dtype=np.float32),
            "actions": np.asarray([self.actions[i] for i in idx], dtype=np.int64),
            "rewards": np.asarray([self.rewards[i] for i in idx], dtype=np.float32),
            "next_states": np.asarray([self.next_states[i] for i in idx], dtype=np.float32),
            "dones": np.asarray([self.dones[i] for i in idx], dtype=np.float32),
            "next_action_masks": np.asarray([self.next_action_masks[i] for i in idx], dtype=np.float32),
        }

    def state_dict(self):
        return {
            "capacity": int(self.capacity),
            "states": list(self.states),
            "actions": list(self.actions),
            "rewards": list(self.rewards),
            "next_states": list(self.next_states),
            "dones": list(self.dones),
            "next_action_masks": list(self.next_action_masks),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        capacity = int(state.get("capacity", self.capacity))
        self.capacity = capacity
        self.states = deque(state.get("states", []), maxlen=capacity)
        self.actions = deque(state.get("actions", []), maxlen=capacity)
        self.rewards = deque(state.get("rewards", []), maxlen=capacity)
        self.next_states = deque(state.get("next_states", []), maxlen=capacity)
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
        hidden_dim = int(policy_cfg["hidden_dim"])
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

        self.team_size = int(team_size)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.device = device
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

        self.online_q = QNet(self.state_dim, int(hidden_dim), self.action_dim).to(self.device)
        self.target_q = QNet(self.state_dim, int(hidden_dim), self.action_dim).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=self.q_base_lr)
        self.replay = ReplayBuffer(replay_size)

        self.total_env_steps = 0
        self.total_update_steps = 0

        # DDQN 只绘制/保存公共训练指标
        self.plot_metric_keys = [
            "Total_Reward_per_Episode",
            "Episode_Length_per_Episode",
        ]
        self.extra_log_metrics = []
        self.extra_save_metrics = []

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
        torch.save(self.online_q.state_dict(), os.path.join(path, "online_q.pth"))
        torch.save(self.target_q.state_dict(), os.path.join(path, "target_q.pth"))

    def load_model(self, path=None):
        if path is None:
            path = self.weights_subdir
        online_path = os.path.join(path, "online_q.pth")
        target_path = os.path.join(path, "target_q.pth")
        if os.path.exists(online_path):
            self.online_q.load_state_dict(torch.load(online_path, map_location=self.device))
        if os.path.exists(target_path):
            self.target_q.load_state_dict(torch.load(target_path, map_location=self.device))

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
        states_np = np.asarray(state_all_agent, dtype=np.float32)
        states_t = torch.as_tensor(states_np, device=self.device)
        with torch.no_grad():
            q_values = self.online_q(states_t).detach().cpu().numpy()

        env = env_core.CURRENT_ENV
        if env is None:
            raise RuntimeError("No active EnvCore instance registered (env_core.CURRENT_ENV is None).")
        masks = np.asarray(env.get_action_masks(), dtype=np.float32)

        eps = self._epsilon()
        actions = []
        for i in range(self.team_size):
            legal = np.flatnonzero(masks[i] > 0.0)
            if legal.size == 0:
                raise RuntimeError(f"No legal actions for agent {i}.")
            if np.random.rand() < eps:
                act = int(np.random.choice(legal))
            else:
                legal_q = q_values[i, legal]
                act = int(legal[int(np.argmax(legal_q))])
            actions.append(act)

        self.total_env_steps += 1
        # 为兼容 train_policy 现有 buffer 字段，返回当前动作掩码
        return actions, [masks[i].copy() for i in range(self.team_size)]

    def _append_rollout_to_replay(self, transition_dicts):
        for i in range(self.team_size):
            agent_traj = transition_dicts[i]
            T = len(agent_traj["states"])
            next_masks = agent_traj.get("next_action_masks", None)
            for t in range(T):
                next_mask = (
                    np.asarray(next_masks[t], dtype=np.float32)
                    if next_masks is not None
                    else np.ones(self.action_dim, dtype=np.float32)
                )
                self.replay.add(
                    state=np.asarray(agent_traj["states"][t], dtype=np.float32),
                    action=int(agent_traj["actions"][t]),
                    reward=float(agent_traj["rewards"][t]),
                    next_state=np.asarray(agent_traj["next_states"][t], dtype=np.float32),
                    done=float(agent_traj["dones"][t]),
                    next_action_mask=next_mask,
                )

    def update(self, transition_dicts, minibatch_size, epoch_num, episode):
        self._apply_lr_schedule(episode)
        self._append_rollout_to_replay(transition_dicts)

        if len(self.replay) < max(self.min_replay_size, 2):
            # 对齐 train_policy 的返回格式：
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

            states = torch.as_tensor(batch["states"], device=self.device)
            actions = torch.as_tensor(batch["actions"], device=self.device).long().unsqueeze(-1)
            rewards = torch.as_tensor(batch["rewards"], device=self.device).unsqueeze(-1)
            next_states = torch.as_tensor(batch["next_states"], device=self.device)
            dones = torch.as_tensor(batch["dones"], device=self.device).unsqueeze(-1)
            next_action_masks = torch.as_tensor(batch["next_action_masks"], device=self.device)

            q_pred = self.online_q(states).gather(1, actions)

            with torch.no_grad():
                online_next_q = self.online_q(next_states)
                masked_online_next_q = online_next_q.masked_fill(next_action_masks <= 0.0, -1e9)
                next_actions = masked_online_next_q.argmax(dim=1, keepdim=True)

                target_next_q = self.target_q(next_states).gather(1, next_actions)
                q_target = rewards + self.gamma * (1.0 - dones) * target_next_q

            loss = F.mse_loss(q_pred, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            self.total_update_steps += 1
            if self.total_update_steps % self.target_update_interval == 0:
                self.target_q.load_state_dict(self.online_q.state_dict())

        # 对齐 train_policy 的返回格式：把 Q-loss 放在 value_loss 位置，
        # alpha 相关指标对 DDQN 不适用，返回 0.0 占位。
        mean_loss = float(np.mean(losses)) if losses else 0.0
        return 0.0, mean_loss, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
