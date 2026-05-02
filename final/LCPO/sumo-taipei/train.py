"""
sumo-taipei/train.py
LCPO 訓練主程式（對應 windy-gym/train.py 的架構）

使用方式：
    cd sumo-taipei/
    python train.py --route_pattern morning_rush --num_epochs 200

流程：
  1. 建立 TaipeiIntersectionEnv（SUMO 交叉口環境）
  2. 使用 LCPO agent（從 windy-gym 借用 core_alg）
  3. 交替執行「早尖峰」和「晚尖峰」以模擬 non-stationary context
"""
import sys
import os
from pathlib import Path

# ── 將 windy-gym agent 加入 PYTHONPATH（借用 LCPO 演算法核心）─────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
WINDY_GYM = REPO_ROOT / "windy-gym"
sys.path.insert(0, str(WINDY_GYM))

from param import get_args
from env import TaipeiIntersectionEnv
from demand.generate_routes import generate_routes, DEMAND_PATTERNS

import numpy as np
import torch

# 使用 windy-gym 的 LCPO agent core
from agent.lcpo import TrainerNet


def make_env(pattern: str, args) -> TaipeiIntersectionEnv:
    route_file = generate_routes(
        pattern_name=pattern,
        sim_duration=args.sim_duration,
        seed=args.seed,
    )
    env = TaipeiIntersectionEnv(
        route_file=route_file,
        sim_duration=args.sim_duration,
        use_gui=args.use_gui,
        lcpo_thresh=args.lcpo_thresh,
        sumo_port=args.sumo_port,
    )
    return env


def train():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 交替 2 種情境模擬 non-stationarity（早尖峰 ↔ 晚尖峰）
    patterns = ["morning_rush", "evening_rush"]

    # 先建立初始環境
    env = make_env(patterns[0], args)
    obs, _ = env.reset(seed=args.seed)

    print(f"[train] Observation dim : {env.observation_space.shape}")
    print(f"[train] Action dim      : {env.action_space.n}")
    print(f"[train] Context dim     : {env.context_size}")

    # ── 建立 LCPO Agent ───────────────────────────────────────────────────────
    # TrainerNet 需要一個 Gymnasium 環境物件
    agent = TrainerNet(
        environment=env,
        lr_rate=args.lr_rate,
        gamma=args.gamma,
        lam=args.lam,
        ood_mini_len=256,
        ood_len=4096,
        trpo_kl_in=args.trpo_kl_in,
        trpo_kl_out=args.trpo_kl_out,
        trpo_damping=args.trpo_damping,
        trpo_dual=args.trpo_dual,
        ood_subsample=args.ood_subsample,
    )

    best_reward = -np.inf
    epoch_rewards = []

    for epoch in range(args.num_epochs):
        # 每隔一定 epoch 切換交通情境（模擬 non-stationarity）
        pattern = patterns[epoch % len(patterns)]
        env.close()
        env = make_env(pattern, args)

        obs, _ = env.reset(seed=args.seed + epoch)
        ep_reward = 0.0
        transitions = []

        # ── 收集一個 batch 的 transitions ─────────────────────────────────
        while len(transitions) < args.master_batch:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward

            transitions.append((obs, action, reward, terminated or truncated))
            obs = next_obs

            if terminated or truncated:
                obs, _ = env.reset()

        # ── 訓練 ──────────────────────────────────────────────────────────
        loss_info = agent.train(transitions)
        epoch_rewards.append(ep_reward)

        avg_reward = np.mean(epoch_rewards[-10:]) if len(epoch_rewards) >= 10 else ep_reward
        print(
            f"[Epoch {epoch:4d}] pattern={pattern:15s} "
            f"reward={ep_reward:8.2f}  avg10={avg_reward:8.2f}"
        )

        # ── 儲存最佳模型 ──────────────────────────────────────────────────
        if epoch % args.save_interval == 0 and avg_reward > best_reward:
            best_reward = avg_reward
            save_path = os.path.join(args.output_dir, f"lcpo_taipei_ep{epoch}.pt")
            agent.save(save_path)
            print(f"  [saved] → {save_path}")

    env.close()
    print(f"\n[train] 完成！最佳平均 reward = {best_reward:.2f}")


if __name__ == "__main__":
    train()
