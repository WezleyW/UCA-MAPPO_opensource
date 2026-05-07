import argparse
import importlib
import os
import shutil
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from env_core import EnvCore
import env_core


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate trained policies.")
    parser.add_argument(
        "--test-config",
        default=os.path.join("configs", "test.yaml"),
        help="Path to test config yaml.",
    )
    parser.add_argument(
        "--scene-config",
        default=None,
        help="Optional scene config path override (higher priority than test.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override (higher priority than test.yaml).",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML file must contain a mapping(dict): {path}")
    return cfg


def setup_seed(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def sanitize_tag(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def resolve_train_cfg(method: str, timestamp: str) -> dict:
    run_cfg_path = os.path.join("output_train", method, timestamp, "configs", "train_config_used.yaml")
    if os.path.exists(run_cfg_path):
        return load_yaml(run_cfg_path)

    fallback_cfg_path = os.path.join("configs", "policy", f"{method}.yaml")
    if os.path.exists(fallback_cfg_path):
        return load_yaml(fallback_cfg_path)

    raise FileNotFoundError(
        f"Cannot find train config for method={method}, timestamp={timestamp}. "
        f"Tried: {run_cfg_path} and {fallback_cfg_path}"
    )


def resolve_model_dir(method: str, timestamp: str, model: str) -> str:
    model = str(model).strip().lower()
    if model not in {"best", "latest"}:
        raise ValueError(f"model must be 'best' or 'latest', got: {model}")

    run_dir = os.path.join("output_train", method, timestamp)
    weights_root = os.path.join(run_dir, f"{method}_weights")
    model_dir = os.path.join(weights_root, f"{model}_model")
    if os.path.isdir(model_dir):
        return model_dir

    # Backward compatibility for older runs without latest_model/best_model folders.
    if model == "latest" and os.path.isdir(weights_root):
        return weights_root

    raise FileNotFoundError(
        f"Cannot find model directory for method={method}, timestamp={timestamp}, model={model}. "
        f"Tried: {model_dir}"
    )


def build_policy(method: str, run_cfg: dict, scene_cfg: dict):
    device = torch.device("cpu")
    training_cfg = run_cfg.get("training", {})
    if isinstance(training_cfg, dict) and "device" in training_cfg:
        try:
            device = torch.device(str(training_cfg["device"]))
        except Exception:
            device = torch.device("cpu")

    env = EnvCore.from_config(scene_cfg=scene_cfg)
    env_core.CURRENT_ENV = env

    policy_cls = importlib.import_module(f"UAV_policy.{method}").UAV_Policy
    policy = policy_cls.from_config(
        run_cfg=run_cfg,
        method=method,
        team_size=env.agent_num,
        state_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    return env, policy


def set_eval_mode(policy):
    for name in [
        "actor",
        "critic",
        "online_q",
        "target_q",
        "agent_q",
        "target_agent_q",
        "mixer",
        "target_mixer",
    ]:
        module = getattr(policy, name, None)
        if isinstance(module, torch.nn.Module):
            module.eval()


def disable_exploration_for_eval(policy):
    # Safety net for value-based policies in case evaluation falls back to policy.take_action().
    if hasattr(policy, "epsilon_start"):
        policy.epsilon_start = 0.0
    if hasattr(policy, "epsilon_end"):
        policy.epsilon_end = 0.0


@torch.no_grad()
def take_greedy_action(policy, state_all_agent):
    env = env_core.CURRENT_ENV
    if env is None:
        raise RuntimeError("No active EnvCore instance registered (env_core.CURRENT_ENV is None).")

    masks_np = np.asarray(env.get_action_masks(), dtype=np.float32)
    device = getattr(policy, "device", torch.device("cpu"))

    actor = getattr(policy, "actor", None)
    if isinstance(actor, torch.nn.Module):
        states_t = torch.as_tensor(np.asarray(state_all_agent, dtype=np.float32), device=device)
        probs = actor(states_t)
        masks_t = torch.as_tensor(masks_np, dtype=torch.float32, device=device)

        masked = probs * masks_t
        row_sums = masked.sum(dim=1, keepdim=True)
        zero_rows = row_sums <= 0.0
        if zero_rows.any():
            allowed = masks_t.sum(dim=1, keepdim=True).clamp(min=1.0)
            safe_uniform = masks_t / allowed
            masked = torch.where(zero_rows, safe_uniform, masked)
            row_sums = masked.sum(dim=1, keepdim=True)
        masked = masked / row_sums.clamp_min(1e-8)
        return masked.argmax(dim=1).detach().cpu().tolist()

    online_q = getattr(policy, "online_q", None)
    if isinstance(online_q, torch.nn.Module):
        states_t = torch.as_tensor(np.asarray(state_all_agent, dtype=np.float32), device=device)
        q_values = online_q(states_t).detach().cpu().numpy()
        actions = []
        for i in range(q_values.shape[0]):
            legal = np.flatnonzero(masks_np[i] > 0.0)
            if legal.size == 0:
                raise RuntimeError(f"No legal actions for agent {i}.")
            actions.append(int(legal[int(np.argmax(q_values[i, legal]))]))
        return actions

    agent_q = getattr(policy, "agent_q", None)
    if isinstance(agent_q, torch.nn.Module):
        obs_t = torch.as_tensor(np.asarray(state_all_agent, dtype=np.float32), device=device)
        q_values = agent_q(obs_t).detach().cpu().numpy()
        actions = []
        for i in range(q_values.shape[0]):
            legal = np.flatnonzero(masks_np[i] > 0.0)
            if legal.size == 0:
                raise RuntimeError(f"No legal actions for agent {i}.")
            actions.append(int(legal[int(np.argmax(q_values[i, legal]))]))
        return actions

    actions, _ = policy.take_action(state_all_agent)
    return actions


@torch.no_grad()
def run_one_episode(env, policy) -> int:
    s = env.reset()
    done_all = False
    steps = 0
    while not done_all:
        actions = take_greedy_action(policy, s)
        s, _, done = env.step(actions)
        steps += 1
        done_all = all(done)
    return int(steps)


def evaluate_policy(env, policy, num_episodes: int) -> np.ndarray:
    out = []
    for _ in range(int(num_episodes)):
        out.append(run_one_episode(env, policy))
    return np.asarray(out, dtype=np.int32)


def plot_distribution(data: np.ndarray, title: str, save_path: str, bins="auto"):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, density=True, alpha=0.75, edgecolor="black")
    plt.title(title)
    plt.xlabel("Episode steps")
    plt.ylabel("Probability density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_overlay_distribution(results: Dict[str, np.ndarray], save_path: str, bins="auto"):
    if not results:
        return
    if len(results) == 1:
        only_key = list(results.keys())[0]
        plot_distribution(
            results[only_key],
            title=f"{only_key} Steps Distribution",
            save_path=save_path,
            bins=bins,
        )
        return

    plt.figure(figsize=(10, 6))
    all_data = np.concatenate(list(results.values()))
    if bins == "auto":
        bins = np.histogram_bin_edges(all_data, bins="auto")

    for label, data in results.items():
        plt.hist(data, bins=bins, density=True, alpha=0.45, edgecolor="black", label=label)

    plt.title("Steps Distribution Overlay")
    plt.xlabel("Episode steps")
    plt.ylabel("Probability density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def ensure_list_of_tests(tests_obj) -> List[dict]:
    if not isinstance(tests_obj, list) or len(tests_obj) == 0:
        raise ValueError("test config field 'tests' must be a non-empty list.")
    for i, item in enumerate(tests_obj):
        if not isinstance(item, dict):
            raise ValueError(f"tests[{i}] must be a mapping(dict).")
        for key in ["method", "timestamp", "model"]:
            if key not in item:
                raise KeyError(f"tests[{i}] missing required key: {key}")
    return tests_obj


def main():
    args = parse_args()
    test_cfg = load_yaml(args.test_config)

    scene_cfg_path = args.scene_config or str(test_cfg.get("scene_config", os.path.join("configs", "scene.yaml")))
    output_dir = args.output_dir or str(test_cfg.get("output_dir", "output_test"))
    num_episodes = int(test_cfg.get("num_episodes", 1000))
    seed = int(test_cfg.get("seed", 42))
    tests = ensure_list_of_tests(test_cfg.get("tests", []))

    scene_cfg = load_yaml(scene_cfg_path)
    setup_seed(seed)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "test_log.txt")

    summary_rows = []
    all_results = {}

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"scene_config={scene_cfg_path}\n")
        logf.write(f"num_episodes={num_episodes}, seed={seed}\n")
        logf.write(f"num_models={len(tests)}\n\n")

        for idx, t in enumerate(tests, start=1):
            method = str(t["method"]).strip()
            timestamp = str(t["timestamp"]).strip()
            model = str(t["model"]).strip().lower()
            tag = sanitize_tag(f"{method}_{timestamp}_{model}")

            logf.write(f"[{idx}/{len(tests)}] Evaluating {tag}\n")
            print(f"[{idx}/{len(tests)}] Evaluating {tag}")

            train_cfg = resolve_train_cfg(method, timestamp)
            env, policy = build_policy(method, train_cfg, scene_cfg)
            model_dir = resolve_model_dir(method, timestamp, model)
            policy.load_model(path=model_dir)
            set_eval_mode(policy)
            disable_exploration_for_eval(policy)

            steps_arr = evaluate_policy(env, policy, num_episodes=num_episodes)
            all_results[tag] = steps_arr

            mean_v = float(np.mean(steps_arr))
            var_v = float(np.var(steps_arr))
            std_v = float(np.std(steps_arr))
            min_v = int(np.min(steps_arr))
            max_v = int(np.max(steps_arr))

            summary_rows.append(
                {
                    "tag": tag,
                    "method": method,
                    "timestamp": timestamp,
                    "model": model,
                    "num_episodes": int(num_episodes),
                    "mean_steps": mean_v,
                    "var_steps": var_v,
                    "std_steps": std_v,
                    "min_steps": min_v,
                    "max_steps": max_v,
                    "model_dir": model_dir,
                }
            )

            np.save(os.path.join(output_dir, f"{tag}_steps.npy"), steps_arr)
            pd.DataFrame(
                {
                    "episode": np.arange(1, len(steps_arr) + 1, dtype=np.int32),
                    "episode_length": steps_arr,
                }
            ).to_csv(os.path.join(output_dir, f"{tag}_steps.csv"), index=False, encoding="utf-8-sig")

            plot_distribution(
                steps_arr,
                title=f"{tag} Steps Distribution (mean={mean_v:.1f}, var={var_v:.1f})",
                save_path=os.path.join(plots_dir, f"{tag}_steps_hist.png"),
            )

            logf.write(
                f"  mean={mean_v:.3f}, var={var_v:.3f}, std={std_v:.3f}, "
                f"min={min_v}, max={max_v}, model_dir={model_dir}\n\n"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_excel(os.path.join(output_dir, "summary.xlsx"), index=False)

    plot_overlay_distribution(
        all_results,
        save_path=os.path.join(plots_dir, "steps_hist_overlay.png"),
        bins="auto",
    )

    print(f"Done. Logs and plots saved under: {output_dir}")


if __name__ == "__main__":
    main()
