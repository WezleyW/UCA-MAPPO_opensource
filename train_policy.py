"""
基于 MARL 的 Multi-UAV 联合搜查任务训练入口
支持：
1) 常规训练
2) 断点续训（resume + resume_timestamp）
"""

import argparse
import importlib
import os
import random

import numpy as np
import torch
import yaml
from tqdm import tqdm
from datetime import datetime
from env_core import EnvCore
from plot_log_save import (
    init_outputs,
    log_best_model_updated,
    log_latest_model_saved,
    log_resume_checkpoint,
    log_training_progress,
    plot_metrics,
    save_metrics_files,
    save_resume_state,
)

def parse_args():
    """
    创建解析命令行参数

    input: none
    output: args （返回 argparse.Namespace，是一种装参数结果的对象，可以通过点操作符访问参数值，如 args.train_config）
    """
    parser = argparse.ArgumentParser(description="Train MARL-based multi-UAV cooperative reconnaissance search policy.")
    parser.add_argument("--train-config", required=True, help="Path to training config yaml.")
    parser.add_argument("--scene-config", required=True, help="Path to scene config yaml.")
    return parser.parse_args()

def resolve_timestamp(run_cfg, training_cfg):
    """
    解析训练时间戳，以及是否使用 resume 模式。如断点续训，则使用断点时间戳，在原断点文件夹继续训练；

    input: run_cfg, training_cfg
    output: timestamp (str), resume (bool)
    """
    resume = training_cfg.get("resume", False)
    if resume:
        timestamp = str(training_cfg.get("resume_timestamp", "")).strip()
        return timestamp, True   
    timestamp = str(run_cfg.get("timestamp")).strip()
    if timestamp.startswith("${now:") and timestamp.endswith("}"):
        return datetime.now().strftime(timestamp[6:-1]), False
    return timestamp, False

def set_seed(seed):
    """
    设置随机种子，确保训练可复现

    input: seed (int)
    output: none (直接设置随机种子)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 1. 解析命令行参数，得到训练配置文件路径和场景配置文件路径
    args = parse_args()                                       # 解析命令行参数，得到 args.train_config 和 args.scene_config 两个路径
    with open(args.train_config, "r", encoding="utf-8") as f: # 打开训练配置文件，其中 f 为文件对象
        run_cfg = yaml.safe_load(f)                           # 解析 yaml 文件，得到 run_cfg 字典，包含训练配置参数
    with open(args.scene_config, "r", encoding="utf-8") as f: # 打开训练配置文件，其中 f 为文件对象
        scene_cfg = yaml.safe_load(f)                         # 解析 yaml 文件，得到 scene_cfg 字典，包含场景配置参数

    # 2. 解析运行参数
    # 2.1 解析单次运行参数
    training_cfg = run_cfg["training"]
    method = str(run_cfg["method"]).strip()                                     # 解析训练方法
    timestamp, resume = resolve_timestamp(run_cfg, training_cfg)                # 解析训练时间戳及是否续训
    PolicyClass = importlib.import_module(f"UAV_policy.{method}").UAV_Policy    # 根据训练方法导入对应的 UAV_Policy 类

    # 2.2 解析训练参数
    plog_interval = int(training_cfg["plog_interval"])
    plog_window = int(training_cfg["plog_window"])
    save_interval_raw = training_cfg.get("save_interval")
    save_interval = None if save_interval_raw is None else int(save_interval_raw)
    total_episodes = int(training_cfg["total_episodes"])
    device = torch.device(str(training_cfg["device"]))
    minibatch_size = int(training_cfg["minibatch_size"])
    epoch_num = int(training_cfg["epoch_num"])
    seed = int(training_cfg["seed"])
    set_seed(seed) 
    output_paths = init_outputs(method, timestamp, run_cfg, scene_cfg, resume=resume) # 初始化输出目录结构，保存相关快照

    # 3. 通过 from_config 创建环境实例，并解析环境参数
    env = EnvCore.from_config(scene_cfg=scene_cfg)
    team_size = env.agent_num
    state_dim = env.obs_dim
    action_dim = env.action_dim

    # 4. 创建策略实例
    UAV_policy = PolicyClass.from_config(
        run_cfg=run_cfg,
        method=method,
        team_size=team_size,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    # 5. 初始化统计指标
    total_rewards_per_episode = []  # 每 episode 的总奖励列表，长度为 episode 数量，每个元素是对应 episode 的总奖励
    episode_lengths = []            # 每 episode 的长度列表，长度为 episode 数量，每个元素是对应 episode 的步数

    policy_losses = []              # 每 episode 的 policy loss 列表，长度为 episode 数量
    value_losses = []               # 每 episode 的 value loss 列表，长度为 episode 数量
    
    entropies = []                  # 每 episode 的 entropy 列表，长度为 episode 数量，策略熵衡量策略随机性
    kls = []                        # 每 episode 的 KL 散度列表，长度为 episode 数量，衡量新旧策略分布的差异
    clip_fracs = []                 # 每 episode 的 clip fraction 列表，长度为 episode 数量，衡量 PPO 中被 clip 掉的比例

    alpha_t_means = []
    alpha_t_stds = []
    alpha_t_mins = []
    alpha_t_maxs = []

    best_avg_episode_length_50 = float("inf") # 50 episode 滑动平均 episode length 的历史最优
    start_episode = 1

    # 6. 断点续训逻辑
    if resume:
        # 6.1 检查续训目录和文件是否存在
        if not os.path.isdir(output_paths["results_dir"]):
            raise FileNotFoundError(f"Resume failed: results dir not found: {output_paths['results_dir']}")
        
        # 6.2 加载 latest model 权重
        if os.path.isdir(output_paths["latest_model_dir"]):
            UAV_policy.load_model(path=output_paths["latest_model_dir"])
        else:
            raise FileNotFoundError(f"Resume failed: latest model dir not found: {output_paths['latest_model_dir']}")

        # 6.3 加载 latest training state（包含训练统计指标、训练进度、随机数状态等）
        # 1）检查训练状态文件是否存在并加载 checkpoint 字典
        if not os.path.exists(output_paths["latest_train_state_path"]):
            raise FileNotFoundError(
                f"Resume failed: training state checkpoint not found: {output_paths['latest_train_state_path']}"
            )
        checkpoint = torch.load(output_paths["latest_train_state_path"], map_location="cpu", weights_only=False) # 返回训练状态字典

        # 2）加载训练统计指标
        metrics = checkpoint.get("metrics", {})
        total_rewards_per_episode = list(metrics.get("total_reward", []))
        episode_lengths = list(metrics.get("episode_length", []))
        policy_losses = list(metrics.get("policy_loss", []))
        value_losses = list(metrics.get("value_loss", []))
        entropies = list(metrics.get("entropy", []))
        kls = list(metrics.get("kl", []))
        clip_fracs = list(metrics.get("clip_frac", []))
        alpha_t_means = list(metrics.get("alpha_t_mean", []))
        alpha_t_stds = list(metrics.get("alpha_t_std", []))
        alpha_t_mins = list(metrics.get("alpha_t_min", []))
        alpha_t_maxs = list(metrics.get("alpha_t_max", []))

        # 3）加载训练状态
        checkpoint_episode = int(checkpoint.get("last_episode"))
        start_episode = max(checkpoint_episode + 1, 1)
        best_avg_episode_length_50 = float(checkpoint.get("best_avg_episode_length_50"))
        policy_state = checkpoint.get("policy_state", None)

        # 4） 加载随机状态
        seed_resume = checkpoint.get("rng_state", None)
        if isinstance(seed_resume, dict):
            try:
                if "python" in seed_resume:
                    random.setstate(seed_resume["python"])
                if "numpy" in seed_resume:
                    np.random.set_state(seed_resume["numpy"])
                if "torch" in seed_resume:
                    torch.set_rng_state(seed_resume["torch"])
                if torch.cuda.is_available() and "torch_cuda" in seed_resume:
                    torch.cuda.set_rng_state_all(seed_resume["torch_cuda"])
            except Exception:
                pass
        
        # 5）日志记录续训信息
        log_resume_checkpoint(output_paths["log_dir"], timestamp, checkpoint_episode, start_episode)

        # 6）重绘历史曲线（使用断点前信息）
        plot_metrics(
            UAV_policy,
            total_rewards_per_episode,
            episode_lengths,
            policy_losses,
            value_losses,
            entropies,
            kls,
            clip_fracs,
            checkpoint_episode,
            plog_window,
            output_paths["plot_dir"],
        )

    # 7. 训练主循环
    with tqdm(total=total_episodes - start_episode + 1,desc="Training",ncols=100,dynamic_ncols=False,) as pbar:
        for episode in range(start_episode, total_episodes + 1):
            # 7.1 初始化每个智能体的经验回放缓冲区（完整 episode 中每个智能体的 s,a,r,s'等）
            buffers = [
                {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                    "action_probs": [],
                    "next_action_masks": [],
                }
                for _ in range(team_size)
            ]

            # 7.2 每 episode 初始化环境
            s = env.reset()
            terminal = False
            episode_total_reward = 0.0
            episode_total_steps = 0

            # 7.3 智能体与环境交互，收集经验数据
            while not terminal:
                episode_total_steps += 1
                actions, prob_dists = UAV_policy.take_action(s)
                next_s, r, done = env.step(actions)
                next_masks = env.get_action_masks()

                step_reward = sum(r)
                episode_total_reward += step_reward

                for i in range(team_size):
                    buffers[i]["states"].append(np.array(s[i]))
                    buffers[i]["actions"].append(actions[i])
                    buffers[i]["next_states"].append(np.array(next_s[i]))
                    buffers[i]["rewards"].append(r[i])
                    buffers[i]["dones"].append(float(done[i]))
                    buffers[i]["action_probs"].append(prob_dists[i])
                    buffers[i]["next_action_masks"].append(np.array(next_masks[i], dtype=np.float32))

                s = next_s
                terminal = all(done)

            # 7.4 使用一条 episode 内数据更新策略，并记录统计指标
            (   a_loss,
                c_loss,
                ent,
                kl,
                clip_frac,
                alpha_t_mean,
                alpha_t_std,
                alpha_t_min,
                alpha_t_max,
            ) = UAV_policy.update(
                buffers, minibatch_size, epoch_num, episode
            )

            total_rewards_per_episode.append(episode_total_reward)
            episode_lengths.append(episode_total_steps)
            policy_losses.append(a_loss)
            value_losses.append(c_loss)
            entropies.append(ent)
            kls.append(kl)
            clip_fracs.append(clip_frac)
            alpha_t_means.append(float(alpha_t_mean))
            alpha_t_stds.append(float(alpha_t_std))
            alpha_t_mins.append(float(alpha_t_min))
            alpha_t_maxs.append(float(alpha_t_max))

            pbar.update(1)

            # 7.5 定期记录训练进度日志、保存模型和统计指标、绘制训练曲线等
            # 1) 每 50 步更新一次训练日志
            if episode % 50 == 0: 
                log_training_progress(
                    UAV_policy,
                    output_paths["log_dir"],
                    episode,
                    total_rewards_per_episode,
                    episode_lengths,
                    policy_losses,
                    value_losses,
                    entropies,
                    kls,
                    clip_fracs,
                    alpha_t_means,
                    alpha_t_stds,
                    alpha_t_mins,
                    alpha_t_maxs,
                    window=50,
                )

            # 2) 若 AvgEpisodeLength(last50) 创历史新低, 则保存 best model
            if episode >= 50:
                avg_episode_length_50 = float(np.mean(episode_lengths[-50:]))
                if avg_episode_length_50 < best_avg_episode_length_50:
                    best_avg_episode_length_50 = avg_episode_length_50
                    UAV_policy.save_model(output_paths["results_dir"], subdir=output_paths["best_model_subdir"])
                    log_best_model_updated(output_paths["log_dir"], episode, avg_episode_length_50)

            # 3) 每 plog_interval 步绘制一次训练曲线
            if episode % plog_interval == 0:
                plot_metrics(
                    UAV_policy,
                    total_rewards_per_episode,
                    episode_lengths,
                    policy_losses,
                    value_losses,
                    entropies,
                    kls,
                    clip_fracs,
                    episode,
                    plog_window,
                    output_paths["plot_dir"],
                )

            # 4) 每 save_interval 步保存一次 latest model 和 checkpoint；
            #    当 save_interval=None 时，关闭周期性 checkpoint，仅在训练结束时保存一次 latest model。
            if save_interval is not None and episode % save_interval == 0:
                # 定期保存 latest model
                UAV_policy.save_model(output_paths["results_dir"], subdir=output_paths["latest_model_subdir"])
                log_latest_model_saved(output_paths["log_dir"], episode)

                # 保存 Excel 版统计指标
                save_metrics_files(
                    UAV_policy,
                    output_paths["results_dir"],
                    total_rewards_per_episode,
                    episode_lengths,
                    policy_losses,
                    value_losses,
                    entropies,
                    kls,
                    clip_fracs,
                    alpha_t_means,
                    alpha_t_stds,
                    alpha_t_mins,
                    alpha_t_maxs,
                )

                # 保存断点续训状态
                save_resume_state(
                    UAV_policy,
                    output_paths["latest_train_state_path"],
                    output_paths["log_dir"],
                    episode,
                    best_avg_episode_length_50,
                    total_rewards_per_episode,
                    episode_lengths,
                    policy_losses,
                    value_losses,
                    entropies,
                    kls,
                    clip_fracs,
                    alpha_t_means,
                    alpha_t_stds,
                    alpha_t_mins,
                    alpha_t_maxs,
                )

    # 8. 训练结束时，若未启用周期性 checkpoint，则仍保存一次 latest model 与统计指标
    if save_interval is None:
        UAV_policy.save_model(output_paths["results_dir"], subdir=output_paths["latest_model_subdir"])
        log_latest_model_saved(output_paths["log_dir"], total_episodes)

        save_metrics_files(
            UAV_policy,
            output_paths["results_dir"],
            total_rewards_per_episode,
            episode_lengths,
            policy_losses,
            value_losses,
            entropies,
            kls,
            clip_fracs,
            alpha_t_means,
            alpha_t_stds,
            alpha_t_mins,
            alpha_t_maxs,
        )

if __name__ == "__main__":
    main()
