"""
TaipeiIntersectionEnv
======================
台北忠孝東路×敦化南路 SUMO 交通號誌控制環境
相容 LCPO（Locally Constrained Policy Optimization）的 Gymnasium wrapper。

設計原則（對應 LCPO 的 WindyGym 介面）：
  - observation  = [state | context]
      state   : 各車道排隊長度、當前 phase、phase 已用時間
      context : 各方向平均車速（non-stationary：隨尖峰/離峰改變）
  - action       : 選擇下一個 phase（離散）
  - reward       : -Σ(等待時間)  （最小化等待時間）
  - is_different : 用 context 的 L2 距離偵測交通情境切換（供 LCPO OOD 偵測）
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 嘗試 import TraCI ──────────────────────────────────────────────────────────
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# ── 路徑常數 ───────────────────────────────────────────────────────────────────
REPO_DIR  = Path(__file__).resolve().parents[1]           # sumo-taipei/
NET_DIR   = REPO_DIR / "network"
DEMAND_DIR = REPO_DIR / "demand"
NET_FILE  = NET_DIR  / "zhongxiao_dunnan.net.xml"
TLS_ID    = "center"

# ── 號誌 Phase 定義 ────────────────────────────────────────────────────────────
# 只控制「綠燈 phase」，黃燈由 SUMO 自動接在後面
# Phase 0：EW 直行綠燈；Phase 2：NS 直行綠燈（對應 tls.xml 中的 phase index）
GREEN_PHASES = [0, 2]
N_PHASES     = len(GREEN_PHASES)  # agent 可選的動作數 = 2

# ── 環境維度 ───────────────────────────────────────────────────────────────────
# state dims:
#   排隊長度 × 8 進入車道（EW 各4 + NS 各3，取4方向各最大車道）
#   = 4 edges × 最多4車道 → 用 4 edges，每 edge 取各車道均值，共 4
#   + 當前 phase one-hot (N_PHASES)
#   + phase 已用時間（正規化）  (1)
STATE_DIM   = 4 + N_PHASES + 1   # = 7
# context dims:
#   4 個進入方向的平均車速（反映交通密度）
CONTEXT_DIM = 4
OBS_DIM     = STATE_DIM + CONTEXT_DIM  # = 11

# 各進入 edge id
IN_EDGES  = ["E_to_C", "W_to_C", "N_to_C", "S_to_C"]

# 最大等候時間（秒），用來正規化
MAX_WAIT   = 300.0
MAX_PHASE_DUR = 120.0
MAX_SPEED  = 13.89  # m/s (= 50 km/h)

# 預設模擬步長（秒）
STEP_LENGTH = 5      # 每個 RL step = 5 秒模擬時間
MIN_GREEN   = 10     # 最短綠燈時間（秒）
MAX_GREEN   = 90     # 最長綠燈時間（秒）


class TaipeiIntersectionEnv(gym.Env):
    """
    台北忠孝東路×敦化南路 SUMO 號誌控制環境（LCPO 相容版）

    Parameters
    ----------
    route_file   : .rou.xml 交通流量檔案路徑
    net_file     : .net.xml 路網檔案路徑（預設自動尋找）
    sim_duration : 每 episode 模擬時間（秒）
    step_length  : SUMO 模擬步長（秒）
    use_gui      : 是否開啟 SUMO-GUI（debug 用）
    lcpo_thresh  : is_different() 的 L2 距離門檻
    sumo_port    : TraCI 連線 port（多環境平行時需不同 port）
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        route_file: str | Path,
        net_file: str | Path = NET_FILE,
        sim_duration: int = 3600,
        step_length: int = STEP_LENGTH,
        use_gui: bool = False,
        lcpo_thresh: float = 0.5,
        sumo_port: int = 8813,
    ):
        super().__init__()
        if not TRACI_AVAILABLE:
            raise ImportError(
                "traci 未安裝。請執行：pip install eclipse-sumo sumo-rl"
            )

        self.route_file   = Path(route_file)
        self.net_file     = Path(net_file)
        self.sim_duration = sim_duration
        self.step_length  = step_length
        self.use_gui      = use_gui
        self.thresh       = lcpo_thresh
        self.port         = sumo_port

        # ── Gymnasium 介面 ────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # 動作：選擇下一個 green phase（0 或 1）
        self.action_space = spaces.Discrete(N_PHASES)

        # ── 內部狀態 ──────────────────────────────────────────────────────────
        self._step          = 0
        self._current_phase = 0   # index into GREEN_PHASES
        self._phase_timer   = 0   # 當前 phase 已持續幾個 RL step
        self._traci_started = False
        self._last_context  = np.zeros(CONTEXT_DIM, dtype=np.float32)

    # ─────────────────────────── LCPO 必要介面 ────────────────────────────────

    @staticmethod
    def no_context_obs(obs: np.ndarray) -> np.ndarray:
        """取出 state 部分（排除 context）"""
        return obs[..., :STATE_DIM]

    @staticmethod
    def only_context(obs: np.ndarray) -> np.ndarray:
        """取出 context 部分（各方向平均車速）"""
        return obs[..., STATE_DIM:]

    @property
    def context_size(self) -> int:
        return CONTEXT_DIM

    def is_different(self, data: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        OOD 偵測：用 context（交通車速）的 L2 距離判斷交通情境是否切換。
        LCPO 的 OutOfDSampler 會呼叫此函式。

        Parameters
        ----------
        data : (batch, OBS_DIM)
        base : (window, OBS_DIM)

        Returns
        -------
        (batch,) bool  — True 表示 OOD（情境已切換）
        """
        base_ctx  = self.only_context(base)          # (window, CONTEXT_DIM)
        data_ctx  = self.only_context(data)          # (batch,  CONTEXT_DIM)
        mu_base   = np.mean(base_ctx, axis=0)        # (CONTEXT_DIM,)
        dist      = np.sum((data_ctx - mu_base) ** 2, axis=-1)  # (batch,)
        return dist > self.thresh

    # ─────────────────────────── Gymnasium 介面 ───────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._close_sumo()

        self._step          = 0
        self._current_phase = 0
        self._phase_timer   = 0

        self._start_sumo()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        """
        執行一個 RL step：
          1. 若 action 與目前 phase 不同，切換號誌（先跑完黃燈）
          2. 推進 step_length 秒的模擬
          3. 回傳 (obs, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        target_phase = GREEN_PHASES[action]
        current_green = GREEN_PHASES[self._current_phase]

        # 若要切換 phase
        if target_phase != current_green:
            self._switch_phase(action)

        # 推進模擬
        for _ in range(self.step_length):
            if self._get_sim_time() >= self.sim_duration:
                break
            traci.simulationStep()

        self._step      += 1
        self._phase_timer += 1

        obs      = self._get_obs()
        reward   = self._get_reward()
        sim_time = self._get_sim_time()
        terminated = sim_time >= self.sim_duration
        truncated  = False
        info = {
            "sim_time":     sim_time,
            "current_phase": self._current_phase,
            "step":          self._step,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self._close_sumo()

    # ─────────────────────────── 內部輔助函式 ─────────────────────────────────

    def _start_sumo(self):
        """啟動 SUMO（透過 TraCI）"""
        sumo_bin = "sumo-gui" if self.use_gui else "sumo"
        # 找 eclipse-sumo 安裝的執行檔
        sumo_home = os.environ.get("SUMO_HOME", "")
        if not sumo_home:
            # 嘗試從 pip 安裝的路徑尋找
            try:
                import sumo
                sumo_home = os.path.dirname(sumo.__file__)
                os.environ["SUMO_HOME"] = sumo_home
                bin_dir = os.path.join(sumo_home, "bin")
                if bin_dir not in sys.path:
                    sys.path.append(bin_dir)
            except ImportError:
                pass

        cmd = [
            sumo_bin,
            "-n", str(self.net_file),
            "-r", str(self.route_file),
            "--step-length", "1",            # SUMO 內部步長 1 秒
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--collision.action", "none",
            "--time-to-teleport", "300",      # 卡超過 5 分鐘強制傳送
            "--waiting-time-memory", "1000",
            "--begin", "0",
            "--end", str(self.sim_duration + 300),
        ]

        traci.start(cmd, port=self.port)
        self._traci_started = True

        # 設定初始號誌
        traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[self._current_phase])

    def _close_sumo(self):
        if self._traci_started:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_started = False

    def _get_sim_time(self) -> float:
        return traci.simulation.getTime()

    def _switch_phase(self, new_action: int):
        """切換到黃燈，再換到新的綠燈 phase"""
        # SUMO tls.xml：green phase 0 → yellow phase 1 → green phase 2 → yellow 3
        yellow_phase = GREEN_PHASES[self._current_phase] + 1
        traci.trafficlight.setPhase(TLS_ID, yellow_phase)
        # 模擬黃燈時間（4 秒）
        for _ in range(4):
            traci.simulationStep()
        # 切換到新綠燈
        traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[new_action])
        self._current_phase = new_action
        self._phase_timer   = 0

    def _get_queue_lengths(self) -> np.ndarray:
        """取得各進入方向的平均排隊長度（單位：車輛數），正規化"""
        queues = []
        for edge in IN_EDGES:
            try:
                n_lanes = traci.edge.getLaneNumber(edge)
                total = sum(
                    traci.lane.getLastStepHaltingNumber(f"{edge}_{i}")
                    for i in range(n_lanes)
                )
                queues.append(float(total) / max(n_lanes, 1))
            except Exception:
                queues.append(0.0)
        return np.array(queues, dtype=np.float32) / 20.0  # 正規化（最大假設 20 輛）

    def _get_context(self) -> np.ndarray:
        """取得 context：各進入方向的平均車速（反映交通密度）"""
        speeds = []
        for edge in IN_EDGES:
            try:
                spd = traci.edge.getLastStepMeanSpeed(edge)
            except Exception:
                spd = MAX_SPEED
            speeds.append(float(spd) / MAX_SPEED)  # 正規化到 [0, 1]
        ctx = np.array(speeds, dtype=np.float32)
        self._last_context = ctx
        return ctx

    def _get_obs(self) -> np.ndarray:
        """建構觀測向量：[queue(4), phase_onehot(N_PHASES), phase_time(1), context(4)]"""
        queue   = self._get_queue_lengths()               # (4,)
        phase_oh = np.zeros(N_PHASES, dtype=np.float32)
        phase_oh[self._current_phase] = 1.0              # (N_PHASES,)
        phase_t = np.array(
            [min(self._phase_timer * self.step_length, MAX_PHASE_DUR) / MAX_PHASE_DUR],
            dtype=np.float32,
        )                                                 # (1,)
        context = self._get_context()                     # (CONTEXT_DIM,)
        return np.concatenate([queue, phase_oh, phase_t, context])

    def _get_reward(self) -> float:
        """reward = -Σ(各車輛累積等待時間) / (可能的最大值)"""
        total_wait = 0.0
        for edge in IN_EDGES:
            try:
                total_wait += traci.edge.getWaitingTime(edge)
            except Exception:
                pass
        # 正規化：負號（越少等待越好）
        return -total_wait / (len(IN_EDGES) * 100.0)
