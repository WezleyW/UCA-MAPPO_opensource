import os
import torch
import torch.nn.functional as F
import numpy as np
import env_core

# 策略网络(Actor)
#输入：每个agent的状态信息
#输出：9个动作的价值（需softmax后变为动作概率）
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x): # 调用神经网络，根据神经网络输入给出输出
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)

# 全局价值网络(CentralValueNet)
# 输入: 所有智能体的状态拼接 (team_size * state_dim)
# 输出: 集群全局价值（后续赋给每一个agent）
class CentralValueNet(torch.nn.Module):
    def __init__(self, total_state_dim, hidden_dim, team_size):
        super(CentralValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)  # 输出总的价值，后面赋给每个actor

    def forward(self, x): # 调用神经网络，根据神经网络输入给出输出
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)  # [batch, 1]

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

    def __init__(self, run_cfg, method, team_size, state_dim, action_dim, device):
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
        scheduler_warmup_steps = int(policy_cfg.get("scheduler_warmup_steps", 0))
        scheduler_decay_steps = int(policy_cfg.get("scheduler_decay_steps", 0))
        scheduler_decay_rate = float(policy_cfg.get("scheduler_decay_rate", 1.0))
        method_name = str(method)

        self.team_size = int(team_size)
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
        self.extra_log_metrics = ["policy_loss", "value_loss", "entropy", "kl", "clip_frac"]
        self.extra_save_metrics = ["policy_loss", "value_loss", "entropy", "kl", "clip_frac"]

        self.actor_base_lr = float(actor_lr)
        self.critic_base_lr = float(critic_lr)
        self.scheduler_warmup_steps = max(int(scheduler_warmup_steps), 0)
        self.scheduler_decay_steps = max(int(scheduler_decay_steps), 0)
        self.scheduler_decay_rate = float(np.clip(scheduler_decay_rate, 0.0, 1.0))

        # 创建Actor网络，所有agent共享（输入：agent观测信息；输出：9个动作的价值）
        self.actor = PolicyNet(int(state_dim), int(hidden_dim), int(action_dim)).to(device)

        # 创建Critic网络，输出结果赋给每个agent（输入：所有智能体状态拼接；输出：全局价值）
        self.critic = CentralValueNet(int(team_size) * int(state_dim), int(hidden_dim), int(team_size)).to(device)

        # 为Actor网络与Critic网络创建优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_base_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_base_lr)

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
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
        critic_path = os.path.join(path, "critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))

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
        # state_per_agent: list of shape [team_size, state_dim] 所有UAV状态组成的列表
        # actions：每个agent的离散动作ID shape=[team_size]
        # action_probs：每个agent的动作概率分布 shape=[team_size,action_dim]

        # 1) 计算掩码前的概率分布
        s = torch.as_tensor(np.asarray(state_all_agent, dtype=np.float32), device=self.device)
        probs = self.actor(s)  # [team_size, action_dim]

        # 2) 取掩码（从模块属性读取，拿到最新实例）
        env = env_core.CURRENT_ENV
        if env is None:
            raise RuntimeError("No active EnvCore instance registered (env_core.CURRENT_ENV is None).")
        masks_list = env.get_action_masks()  # List[np.ndarray]，len=team_size
        masks = torch.as_tensor(np.asarray(masks_list, dtype=np.float32), device=self.device)

        # 3) 应用掩码并归一化
        masked = probs * masks
        row_sums = masked.sum(dim=1, keepdim=True)
        zero_rows = (row_sums == 0.0)
        if zero_rows.any():
            allowed_counts = masks.sum(dim=1, keepdim=True).clamp(min=1.0)
            masked = torch.where(allowed_counts > 0, masks / allowed_counts, masks)
        else:
            masked = masked / row_sums

        # 4) 采样并返回用于 MAPPO 的概率分布
        action_dist = torch.distributions.Categorical(masked)
        actions_t = action_dist.sample()
        actions = actions_t.detach().cpu().tolist()

        probs_np = masked.detach().cpu().numpy()
        action_probs = [probs_np[i] for i in range(probs_np.shape[0])]
        return actions, action_probs

    def update(self, transition_dicts, minibatch_size, epoch_num, episode):
        self._apply_lr_schedule(episode)
        beta = self.entropy_coef(episode)
        # transition_dicts[i] 为第 i 个智能体的字典（dict[str, list]），list 长度为 T，包含整个 episode 中对应键的信息
        # 1 计算该 episode 的长度，即 rollout 中的数据尺寸
        T = len(transition_dicts[0]['states'])

        # 2 拼接各智能体状态，构建全局状态
        states_global = []
        next_states_global = []
        for t in range(T):
            concat_state = []
            concat_next_state = []
            for i in range(self.team_size):
                concat_state.append(transition_dicts[i]['states'][t])
                concat_next_state.append(transition_dicts[i]['next_states'][t])
            states_global.append(np.concatenate(concat_state))
            next_states_global.append(np.concatenate(concat_next_state))
        # 全局状态：长度为 T 的list，states_global[t] 表示在时间步 t 时的全局状态（ Critic 网络的输入）shape=[T, team_size*state_dim]
        states_global = torch.as_tensor(np.asarray(states_global, dtype=np.float32),
                                          device=self.device)
        # 全局状态：长度为 T 的list，states_global[t] 表示在时间步 t+1 时的全局状态（ Critic 网络的输入）shape=[T, team_size*state_dim]
        next_states_global = torch.as_tensor(np.asarray(next_states_global, dtype=np.float32),
                                          device=self.device)

        # 3 提取所有智能体信息
        # 所有智能体的奖励：长度为 T 的list，reward_all[t] 表示在时间步 t 时所有智能体的奖励，shape=[T, team_size]
        rewards_all = torch.as_tensor(np.asarray([[transition_dicts[i]['rewards'][t] for i in range(self.team_size)]
                                                  for t in range(T)], dtype=np.float32),
                                      device=self.device)
        # 所有智能体的 done：长度为 T 的list，done_all[t] 表示在时间步 t 时所有智能体是否完成任务，shape=[T, team_size]
        dones_all = torch.as_tensor(np.asarray([[transition_dicts[i]['dones'][t] for i in range(self.team_size)]
                                                for t in range(T)], dtype=np.float32),
                                    device=self.device)

        # 4 通过 critic 网络计算 GAE
        values_global = self.critic(states_global).squeeze(-1)             # 通过 critic 网络计算各时间步的全局 V：长度为 T 的 list，shape=[T]
        next_values_global = self.critic(next_states_global).squeeze(-1)   # 通过 critic 网络计算各时间步下一时间步的全局 V：长度为 T 的 list，shape=[T]
        reward_Global = rewards_all.mean(dim=1)                               # 各时间步的奖励均值：长度为 T 的 list，shape=[T]
        Done = dones_all.max(dim=1).values                          # 各时间步的 done：长度为 T 的 list，shape=[T] 0/1
        td_target = reward_Global + self.gamma * next_values_global * (1 - Done)     # 各时间步的 TD-target：长度为 T 的 list，shape=[T]
        td_delta = td_target - values_global                               # 各时间步的 TD-error：长度为 T 的 list，shape=[T]
        adv_global = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)  # 各时间步的优势函数：长度为 T 的 list，shape=[T]
        adv_all = adv_global.unsqueeze(0).expand(self.team_size, -1).contiguous()    # 各时间步所有智能体的优势函数（所有智能体一致），长度为 T 的 list，shape=[T，team_size]
        with torch.no_grad():   # 对所有智能体所有时间步的GAE做归一化
            mean = adv_all.mean()
            std = adv_all.std(unbiased=False).clamp_min(1e-8)
            adv_all_norm = (adv_all - mean) / std  # [N, T] 统一归一化

        critic_losses = []                                          # shape=[每个epoch的minibatch数*epoch数]
        action_losses = []                                          # shape=[每个epoch的minibatch数*epoch数]
        entropies = []                                              # shape=[每个epoch的minibatch数*epoch数]
        kls = []                                                    # shape=[每个epoch的minibatch数*epoch数]
        clip_fracs = []                                             # shape=[每个epoch的minibatch数*epoch数]；记录被剪裁比例（每个 minibatch 的平均）

        for _ in range(epoch_num):                                  # 每个 trajectory 训练 epoch_num 次

            # 5 切分 mini-batch
            idx_all = np.random.permutation(T)                      # 返回一个长度为 T 的数组，为 [0,...,T-1] 的乱序
            mb_size = T if (minibatch_size is None or minibatch_size <= 0) else min(minibatch_size, T)
            num_full = T // mb_size                                 # 能凑出的完整 minibatch 数
            if num_full == 0:                                       # 保险：当 T < mb_size 时仍做一次全量更新
                num_full = 1
                mb_size = T
            idx_full = idx_all[:num_full * mb_size].reshape(num_full, mb_size)  # 形状 [num_full, mb_size]

            # 6 按 mini-batch 多次更新 critic 网络
            for mb_idx_np in idx_full:
                mb_idx = torch.as_tensor(mb_idx_np, device=self.device)

                values_mb = self.critic(states_global[mb_idx]).squeeze(-1)  # minibatch 中 critic 网络预测的 V
                td_target_mb = td_target[mb_idx].detach()                   # minibatch 的 TD-target

                critic_loss = F.mse_loss(values_mb, td_target_mb)  # minibatch 的 MSE 损失
                self.critic_optimizer.zero_grad()                  # 梯度清零
                critic_loss.backward()                             # 计算梯度
                # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 梯度剪裁
                self.critic_optimizer.step()                       # 更新 Critic 网络参数
                critic_losses.append(critic_loss.item())           # 记录 Critic 网络的 loss

            # 6 更新actor网络（带掩码）
            EPS = 1e-8  # 数值稳定
            epoch_kls = []            # 记录本 epoch 内每个 minibatch 的平均 KL
            early_stop_epoch = False  # 若本 epoch 内出现 KL 过大，则跳出本 epoch
            orig_lrs = [g['lr'] for g in self.actor_optimizer.param_groups]

            for mb_idx_np in idx_full:
                mb_idx = torch.as_tensor(mb_idx_np, device=self.device)

                total_actor_loss = 0.0
                batch_action_losses = []    # 该 mini-batch 内所有 UAV 的 actor loss
                batch_entropies = []        # 该 mini-batch 内所有 UAV 的平均策略熵
                batch_kls = []              # 该 mini-batch 内所有 UAV 的 kl
                batch_frac_clipped = []     # 该 mini-batch 内所有 UAV 的剪裁比例

                for i in range(self.team_size):
                   # 6.1 提取智能体 i 一个 episode 完整信息
                    states = torch.as_tensor(np.asarray(transition_dicts[i]['states'], dtype=np.float32),
                                         device=self.device)  # [T, state_dim]
                    actions = torch.as_tensor(transition_dicts[i]['actions'], device=self.device).view(-1, 1)  # [T,1]
                    old_probs = torch.as_tensor(np.asarray(transition_dicts[i]['action_probs'], dtype=np.float32),
                                            device=self.device)  # [T, action_dim]; 这是“已掩码并归一化”的行为策略

                    # 6.2 提取 minibatch 所需信息
                    states_mb = states[mb_idx]
                    actions_mb = actions[mb_idx]
                    old_probs_mb = old_probs[mb_idx]
                    adv_mb = adv_all_norm[i, mb_idx].unsqueeze(-1)   # [mb, 1]

                    # 6.3 反推出每步的掩码（old_probs>0 的即当时允许的动作）
                    mask_mb = (old_probs_mb > 0).to(old_probs_mb.dtype)

                    # 6.4 计算当前策略分布并应用掩码 + 归一化
                    current_probs_raw = self.actor(states_mb)
                    current_probs = current_probs_raw * mask_mb
                    row_sums = current_probs.sum(dim=1, keepdim=True)
                    zero_rows = (row_sums <= 0)
                    if zero_rows.any():
                        allowed_counts = mask_mb.sum(dim=1, keepdim=True).clamp(min=1.0)
                        safe_uniform = mask_mb / allowed_counts
                        current_probs = torch.where(zero_rows, safe_uniform, current_probs)
                        row_sums = current_probs.sum(dim=1, keepdim=True)
                    current_probs = current_probs / row_sums.clamp_min(EPS)

                    # 计算 KL(old || new)；注意 old_probs_mb 已是行为策略(含掩码)的分布
                    # KL = sum_a old * (log old - log new)，对 batch 取均值
                    with torch.no_grad():
                        kl_per_agent = (old_probs_mb.clamp_min(EPS) * (
                                torch.log(old_probs_mb.clamp_min(EPS)) - torch.log(current_probs.clamp_min(EPS))
                        )).sum(dim=1).mean()  # 标量
                        batch_kls.append(kl_per_agent.item())

                    # 6.5 PPO 目标：用“掩码后的当前分布”与“行为分布 old_probs”计算比值
                    cur_sel = current_probs.gather(1, actions_mb).clamp_min(EPS)  # 提取所选动作的 current prob
                    old_sel = old_probs_mb.gather(1, actions_mb).clamp_min(EPS)   # 提取所选动作的 old prob

                    cur_log_probs = torch.log(cur_sel)
                    old_log_probs = torch.log(old_sel).detach()

                    ratio = torch.exp(cur_log_probs - old_log_probs)  # [T,1]
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_mb

                    frac_clip_i = ((ratio > 1.0 + self.eps) | (ratio < 1.0 - self.eps)).float().mean().item()  # 统计该 UAV 在本 minibatch 的“被剪裁比例”
                    batch_frac_clipped.append(frac_clip_i)

                    action_loss = torch.mean(-torch.min(surr1, surr2))  # Clip 后的智能体 i 在当前 mini-batch 的 PPO 策略损失均值，即L=E[...]
                    entropy_val = compute_entropy(current_probs)        # 熵（用于日志）
                    entropy_t = torch.distributions.Categorical(probs=current_probs).entropy().mean() # 熵（用于反传）

                    total_actor_loss = total_actor_loss + (action_loss - beta * entropy_t)   # 当前 mini-batch 内所有智能体 i 的 L=E[...] 求和并熵正则，也是 Actor 网络的优化目标
                    batch_action_losses.append(action_loss.item())                           # 当前 mini-batch、逐智能体的 action_loss 的数值副本列表，shape=[team_size]
                    batch_entropies.append(entropy_val)                                      # 当前 mini-batch、逐智能体的 entropy 的数值副本列表，shape=[team_size]

                # 记录该 mini-batch 的均值
                action_losses.append(float(np.mean(batch_action_losses))) # 跨 mini-batch，epoch 的汇总；每个 epoch，mini-batch 所有 UAV 的 action_losses 的均值、再 append 到这个列表里。
                entropies.append(float(np.mean(batch_entropies)))
                mb_kl = float(np.mean(batch_kls))   # 一个 minibatch 内所有 UAV KL的均值
                clip_fracs.append(float(np.mean(batch_frac_clipped)))
                epoch_kls.append(mb_kl)             # 一个 epoch 内所有 minibatch 的平均 kl
                kls.append(mb_kl)                   # 跨 epoch 的所有 minibatch 的平均 kl

                # 每个 minibatch 结束后，判断是否 kl 过大，若过大则临时降低学习率
                if mb_kl > 2.0 * self.target_kl:                    # 过度超标，直接舍弃使用本 trajectory 更新
                    early_stop_epoch = True
                    break
                elif mb_kl > 1.5 * self.target_kl:                  # 轻微超标：本 minibatch 降低一点 lr 再更新
                    for g in self.actor_optimizer.param_groups:
                        g['lr'] = max(g['lr'] * 0.5, 1e-6)     # 临时降一半并设下限

                # 更新 actor 网络
                total_actor_loss = total_actor_loss / self.team_size  # 对 UAV 求均值
                self.actor_optimizer.zero_grad()    # 把 Actor 网络里所有参数的 .grad 清零。
                total_actor_loss.backward()         # 对标量 total_actor_loss 做反向传播（autograd），即计算梯度
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # 梯度剪裁
                self.actor_optimizer.step()         # 按优化器（Adam、SGD 等）的更新公式实际修改参数值，即更新θ

            # 每个 epoch 结束后，恢复可能临时降低的 actor_lr
            for g, lr in zip(self.actor_optimizer.param_groups, orig_lrs):
                g['lr'] = lr

            # 每个 epoch 结束后，判断是否 KL 早停
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
            0.0,
            0.0,
            0.0,
            0.0,
        )
        # 返回 1.action loss均值（minibatch维，UAV维，K维）；2.critic loss 均值（minibatch维，UAV维，K维）；3.entropy 均值（minibatch维，UAV维，K维）
