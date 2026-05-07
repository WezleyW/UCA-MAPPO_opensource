import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

DEFAULT_PLOT_METRIC_KEYS = [
    "Total_Reward_per_Episode",
    "Episode_Length_per_Episode",
    "Policy_Loss_per_Episode",
    "Value_Loss_per_Episode",
    "Entropy_per_Episode",
    "KL_per_Episode",
    "Clip_frac_per_Episode",
]

DEFAULT_EXTRA_LOG_METRICS = ["policy_loss", "value_loss", "entropy", "kl", "clip_frac"]
DEFAULT_EXTRA_SAVE_METRICS = ["policy_loss", "value_loss", "entropy", "kl", "clip_frac"]


def init_outputs(method, timestamp, run_cfg, scene_cfg, resume=False, base_dir="output_train"):
    results_dir = os.path.join(base_dir, method, timestamp)
    if not resume:
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)
    else:
        os.makedirs(results_dir, exist_ok=True)

    plot_dir = os.path.join(results_dir, f"{method}_plots")
    os.makedirs(plot_dir, exist_ok=True)
    logs_dir = os.path.join(results_dir, f"{method}_logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_dir = os.path.join(logs_dir, f"{method}_logs.txt")
    if not resume:
        open(log_dir, "w", encoding="utf-8").close()
    elif not os.path.exists(log_dir):
        open(log_dir, "w", encoding="utf-8").close()

    weights_dir = f"{method}_weights"
    latest_model_subdir = os.path.join(weights_dir, "latest_model")
    best_model_subdir = os.path.join(weights_dir, "best_model")
    latest_model_dir = os.path.join(results_dir, latest_model_subdir)

    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    latest_train_state_path = os.path.join(checkpoints_dir, "latest_training_state.pt")

    configs_dir = os.path.join(results_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    run_cfg_snapshot = dict(run_cfg)
    run_cfg_snapshot["timestamp"] = timestamp
    train_cfg_snapshot_path = os.path.join(configs_dir, "train_config_used.yaml")
    scene_cfg_snapshot_path = os.path.join(configs_dir, "scene_config_used.yaml")
    if (not resume) or (not os.path.exists(train_cfg_snapshot_path)):
        with open(train_cfg_snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(run_cfg_snapshot, f, allow_unicode=True, sort_keys=False)
    if (not resume) or (not os.path.exists(scene_cfg_snapshot_path)):
        with open(scene_cfg_snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(scene_cfg, f, allow_unicode=True, sort_keys=False)

    return {
        "results_dir": results_dir,
        "plot_dir": plot_dir,
        "log_dir": log_dir,
        "weights_dir": weights_dir,
        "latest_model_subdir": latest_model_subdir,
        "best_model_subdir": best_model_subdir,
        "latest_model_dir": latest_model_dir,
        "checkpoints_dir": checkpoints_dir,
        "latest_train_state_path": latest_train_state_path,
        "configs_dir": configs_dir,
        "train_cfg_snapshot_path": train_cfg_snapshot_path,
        "scene_cfg_snapshot_path": scene_cfg_snapshot_path,
    }


def log_message(log_dir, message):
    with open(log_dir, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def get_metric_config(policy):
    return {
        "plot_metric_keys": getattr(policy, "plot_metric_keys", DEFAULT_PLOT_METRIC_KEYS),
        "extra_log_metrics": getattr(policy, "extra_log_metrics", DEFAULT_EXTRA_LOG_METRICS),
        "extra_save_metrics": getattr(policy, "extra_save_metrics", DEFAULT_EXTRA_SAVE_METRICS),
    }


def log_resume_checkpoint(log_dir, timestamp, last_episode, start_episode):
    log_message(
        log_dir,
        f"Resume from checkpoint: timestamp={timestamp}, last_episode={last_episode}, start_episode={start_episode}",
    )


def log_training_progress(
    policy,
    log_dir,
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
):
    metric_config = get_metric_config(policy)
    extra_log_metrics = metric_config["extra_log_metrics"]

    avg_reward = np.mean(total_rewards_per_episode[-window:])
    avg_length = np.mean(episode_lengths[-window:])
    avg_policy_loss = np.mean(policy_losses[-window:])
    avg_value_loss = np.mean(value_losses[-window:])
    avg_entropy = np.mean(entropies[-window:])
    avg_kl = np.mean(kls[-window:])
    avg_clip_frac = np.mean(clip_fracs[-window:])

    log_parts = [
        f"Episode {episode}",
        f"AvgTotalReward(last{window})={avg_reward:.3f}",
        f"AvgEpisodeLength(last{window})={avg_length:.3f}",
    ]
    if "policy_loss" in extra_log_metrics:
        log_parts.append(f"AvgPolicyLoss(last{window})={avg_policy_loss:.3f}")
    if "value_loss" in extra_log_metrics:
        log_parts.append(f"AvgValueLoss(last{window})={avg_value_loss:.3f}")
    if "entropy" in extra_log_metrics:
        log_parts.append(f"AvgEntropy(last{window})={avg_entropy:.3f}")
    if "kl" in extra_log_metrics:
        log_parts.append(f"AvgKL(last{window})={avg_kl:.3f}")
    if "clip_frac" in extra_log_metrics:
        log_parts.append(f"clip_frac={avg_clip_frac:.3f}")

    if "alpha_t_mean" in extra_log_metrics:
        alpha_mean = float(np.mean(np.asarray(alpha_t_means[-window:], dtype=np.float32)))
        log_parts.append(f"AvgAlphaT(last{window})={alpha_mean:.4f}")
    if "alpha_t_std" in extra_log_metrics:
        alpha_std = float(np.mean(np.asarray(alpha_t_stds[-window:], dtype=np.float32)))
        log_parts.append(f"AlphaTStd(last{window})={alpha_std:.4f}")
    if ("alpha_t_min" in extra_log_metrics) or ("alpha_t_max" in extra_log_metrics):
        alpha_min = float(np.mean(np.asarray(alpha_t_mins[-window:], dtype=np.float32)))
        alpha_max = float(np.mean(np.asarray(alpha_t_maxs[-window:], dtype=np.float32)))
        log_parts.append(f"AlphaTRange(last{window})=[{alpha_min:.4f},{alpha_max:.4f}]")

    log_message(log_dir, ", ".join(log_parts))


def log_best_model_updated(log_dir, episode, avg_episode_length):
    log_message(
        log_dir,
        f"Best model updated at episode {episode}, AvgEpisodeLength(last50)={avg_episode_length:.3f}",
    )


def log_latest_model_saved(log_dir, episode):
    log_message(log_dir, f"Latest model saved at episode {episode}")


def log_training_state_saved(log_dir, episode):
    log_message(log_dir, f"Training state saved at episode {episode}")


def save_resume_state(
    policy,
    latest_train_state_path,
    log_dir,
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
):
    policy_state = {}
    if hasattr(policy, "get_resume_state"):
        policy_state = policy.get_resume_state()

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    train_state = {
        "last_episode": int(episode),
        "best_avg_episode_length_50": float(best_avg_episode_length_50),
        "metrics": {
            "total_reward": total_rewards_per_episode,
            "episode_length": episode_lengths,
            "policy_loss": policy_losses,
            "value_loss": value_losses,
            "entropy": entropies,
            "kl": kls,
            "clip_frac": clip_fracs,
            "alpha_t_mean": alpha_t_means,
            "alpha_t_std": alpha_t_stds,
            "alpha_t_min": alpha_t_mins,
            "alpha_t_max": alpha_t_maxs,
        },
        "policy_state": policy_state,
        "rng_state": rng_state,
    }
    torch.save(train_state, latest_train_state_path)
    log_training_state_saved(log_dir, episode)


def save_metrics_files(
    policy,
    results_dir,
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
):
    metric_config = get_metric_config(policy)
    extra_save_metrics = metric_config["extra_save_metrics"]

    n = min(len(total_rewards_per_episode), len(episode_lengths))
    data = {
        "episode": np.arange(1, n + 1),
        "total_reward": total_rewards_per_episode[:n],
        "episode_length": episode_lengths[:n],
    }
    if "policy_loss" in extra_save_metrics:
        data["policy_loss"] = policy_losses[:n]
    if "value_loss" in extra_save_metrics:
        data["value_loss"] = value_losses[:n]
    if "entropy" in extra_save_metrics:
        data["entropy"] = entropies[:n]
    if "kl" in extra_save_metrics:
        data["kl"] = kls[:n]
    if "clip_frac" in extra_save_metrics:
        data["clip_frac"] = clip_fracs[:n]

    if "alpha_t_mean" in extra_save_metrics:
        data["alpha_t_mean"] = alpha_t_means[:n]
    if "alpha_t_std" in extra_save_metrics:
        data["alpha_t_std"] = alpha_t_stds[:n]
    if "alpha_t_min" in extra_save_metrics:
        data["alpha_t_min"] = alpha_t_mins[:n]
    if "alpha_t_max" in extra_save_metrics:
        data["alpha_t_max"] = alpha_t_maxs[:n]

    df = pd.DataFrame(data)
    csv_path = os.path.join(results_dir, "metrics_per_episode.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    xlsx_path = os.path.join(results_dir, "metrics_per_episode.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_episode", index=False)
    return csv_path, xlsx_path


def _moving_avg(y, window):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0:
        return y
    if window is None or window <= 1:
        return y.copy()

    csum = np.cumsum(y)
    out = np.empty_like(y)
    for t in range(n):
        start = max(0, t - window + 1)
        s = csum[t] - (csum[start - 1] if start > 0 else 0.0)
        out[t] = s / (t - start + 1)
    return out


def _build_all_metrics_dict(
    total_rewards_per_episode,
    episode_lengths,
    policy_losses,
    value_losses,
    entropies,
    kls,
    clip_fracs,
):
    return {
        "Total_Reward_per_Episode": total_rewards_per_episode,
        "Episode_Length_per_Episode": episode_lengths,
        "Policy_Loss_per_Episode": policy_losses,
        "Value_Loss_per_Episode": value_losses,
        "Entropy_per_Episode": entropies,
        "KL_per_Episode": kls,
        "Clip_frac_per_Episode": clip_fracs,
    }


def plot_metrics(
    policy,
    total_rewards_per_episode,
    episode_lengths,
    policy_losses,
    value_losses,
    entropies,
    kls,
    clip_fracs,
    episode,
    window,
    plot_dir,
):
    """
    绘制 7 张指标图到 plot_dir 目录：
      - 浅蓝色折线：每 episode 的原始数据
      - 深蓝色折线：按给定 window 的滑动平均（尾部扩展窗口，前期窗口自适应）
    """
    os.makedirs(plot_dir, exist_ok=True)
    metric_config = get_metric_config(policy)
    plot_metric_keys = metric_config["plot_metric_keys"]
    all_metrics = _build_all_metrics_dict(
        total_rewards_per_episode,
        episode_lengths,
        policy_losses,
        value_losses,
        entropies,
        kls,
        clip_fracs,
    )
    metrics_dict = {k: v for k, v in all_metrics.items() if k in plot_metric_keys}

    metric_meta = {
        "Total_Reward_per_Episode": ("Total Reward", "TotalReward.png"),
        "Episode_Length_per_Episode": ("Episode Length", "EpisodeLength.png"),
        "Policy_Loss_per_Episode": ("Policy Loss", "PolicyLoss.png"),
        "Value_Loss_per_Episode": ("Value Loss", "ValueLoss.png"),
        "Entropy_per_Episode": ("Entropy", "Entropy.png"),
        "KL_per_Episode": ("KL", "KL.png"),
        "Clip_frac_per_Episode": ("Clip fraction", "ClipFrac.png"),
    }

    for key, values in metrics_dict.items():
        y_label, fname = metric_meta.get(key, (key, f"{key}.png"))
        y_raw = np.asarray(values, dtype=np.float64)
        if y_raw.size == 0:
            print(f"[plot] Warning: '{key}' empty, skip.")
            continue

        x = np.arange(1, len(y_raw) + 1)
        y_smooth = _moving_avg(y_raw, int(window) if window is not None else 0)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y_raw, "-", linewidth=1.2, color="lightskyblue", alpha=0.85, label="Raw")
        plt.plot(
            x,
            y_smooth,
            "-",
            linewidth=2.0,
            color="royalblue",
            alpha=1.0,
            label=f"Moving Avg (w={window})" if (window and window > 1) else "Moving Avg (w=1)",
        )

        plt.title(f"{y_label} (up to Episode {episode})")
        plt.xlabel("Episodes")
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.30)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(plot_dir, fname)
        plt.savefig(out_path)
        plt.close()
