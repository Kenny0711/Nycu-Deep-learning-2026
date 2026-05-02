"""
sumo-taipei/param.py
LCPO 訓練超參數（對應 windy-gym/param.py 的風格）
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="LCPO on Taipei Intersection (SUMO)"
    )

    # ── 環境 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--route_pattern", type=str,
                        default="morning_rush",
                        choices=["morning_rush", "off_peak",
                                 "evening_rush", "night"],
                        help="交通流量情境")
    parser.add_argument("--sim_duration", type=int, default=3600,
                        help="每 episode 模擬時間（秒）")
    parser.add_argument("--use_gui", action="store_true",
                        help="開啟 SUMO-GUI（debug 用）")
    parser.add_argument("--sumo_port", type=int, default=8813,
                        help="TraCI 連線 port")

    # ── LCPO OOD 偵測 ─────────────────────────────────────────────────────────
    parser.add_argument("--lcpo_thresh", type=float, default=0.3,
                        help="is_different() L2 距離門檻")

    # ── LCPO / TRPO 參數 ──────────────────────────────────────────────────────
    parser.add_argument("--trpo_kl_in",   type=float, default=0.1)
    parser.add_argument("--trpo_kl_out",  type=float, default=0.001)
    parser.add_argument("--trpo_damping", type=float, default=0.1)
    parser.add_argument("--trpo_dual",    action="store_true")
    parser.add_argument("--ood_subsample",type=int,   default=1)

    # ── 訓練 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--num_epochs",   type=int,   default=500)
    parser.add_argument("--master_batch", type=int,   default=4096)
    parser.add_argument("--lr_rate",      type=float, default=3e-4)
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--lam",          type=float, default=0.95)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output_dir",   type=str,   default="./tests/")
    parser.add_argument("--save_interval",type=int,   default=50)
    parser.add_argument("--eval_interval",type=int,   default=10)

    return parser.parse_args()
