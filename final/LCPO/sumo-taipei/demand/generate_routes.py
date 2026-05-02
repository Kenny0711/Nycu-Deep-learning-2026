"""
台北忠孝東路×敦化南路 交通流量生成器
Generate time-varying traffic demand for LCPO non-stationarity context.

交通量模式（veh/hr per direction）：
  早尖峰  07:00-09:00  EW: 2400, WE: 1800, NS: 1400, SN: 1000
  離峰    09:00-17:00  EW: 1200, WE: 1000, NS:  700, SN:  600
  晚尖峰  17:00-20:00  EW: 2000, WE: 2200, NS: 1600, SN: 1400
  夜間    20:00-07:00  EW:  400, WE:  350, NS:  300, SN:  250

流向比例（turning ratio）：
  直行 70%，左轉 20%，右轉 10%
"""
import os
import numpy as np

# ── 輸出路徑 ───────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 模擬時間（秒）─────────────────────────────────────────────────────────────
SIM_DURATION = 7200   # 2 小時（可調整）
STEP_INTERVAL = 600   # 每 10 分鐘一個需求區間

# ── 交通流量 (veh/hr) ─────────────────────────────────────────────────────────
# 每個時段：(E→W, W→E, N→S, S→N)
DEMAND_PATTERNS = {
    "morning_rush":  (2400, 1800, 1400, 1000),  # 早尖峰
    "off_peak":      (1200, 1000,  700,  600),  # 離峰
    "evening_rush":  (2000, 2200, 1600, 1400),  # 晚尖峰
    "night":         ( 400,  350,  300,  250),  # 夜間
}

# 轉向比例
THRU_RATIO  = 0.70
LEFT_RATIO  = 0.20
RIGHT_RATIO = 0.10


def veh_per_interval(flow_per_hr: float, interval_sec: int) -> int:
    """將 veh/hr 換算成每區間車輛數"""
    return max(1, int(flow_per_hr * interval_sec / 3600))


def generate_routes(pattern_name: str = "morning_rush",
                    sim_duration: int = SIM_DURATION,
                    seed: int = 42) -> str:
    """
    生成 .rou.xml 格式的路徑需求檔案。
    回傳輸出檔案路徑。
    """
    rng = np.random.default_rng(seed)
    ew, we, ns, sn = DEMAND_PATTERNS[pattern_name]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '',
        '    <!-- 車輛類型：一般小客車 -->',
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0"'
        ' maxSpeed="13.89" carFollowModel="IDM"/>',
        '',
        '    <!-- ======= 路徑定義 ======= -->',
        '    <!-- 忠孝東路 直行 -->',
        '    <route id="EW_thru"  edges="E_to_C C_to_W"/>',
        '    <route id="WE_thru"  edges="W_to_C C_to_E"/>',
        '    <!-- 忠孝東路 左轉 -->',
        '    <route id="EW_left"  edges="E_to_C C_to_N"/>',
        '    <route id="WE_left"  edges="W_to_C C_to_S"/>',
        '    <!-- 忠孝東路 右轉 -->',
        '    <route id="EW_right" edges="E_to_C C_to_S"/>',
        '    <route id="WE_right" edges="W_to_C C_to_N"/>',
        '    <!-- 敦化南路 直行 -->',
        '    <route id="NS_thru"  edges="N_to_C C_to_S"/>',
        '    <route id="SN_thru"  edges="S_to_C C_to_N"/>',
        '    <!-- 敦化南路 左轉 -->',
        '    <route id="NS_left"  edges="N_to_C C_to_E"/>',
        '    <route id="SN_left"  edges="S_to_C C_to_W"/>',
        '    <!-- 敦化南路 右轉 -->',
        '    <route id="NS_right" edges="N_to_C C_to_W"/>',
        '    <route id="SN_right" edges="S_to_C C_to_E"/>',
        '',
        '    <!-- ======= 車輛發車 ======= -->',
    ]

    veh_id = 0
    # 按時段逐步發車
    for t_start in range(0, sim_duration, STEP_INTERVAL):
        t_end = min(t_start + STEP_INTERVAL, sim_duration)
        dt = t_end - t_start

        # 每個方向 + 轉向組合
        combos = [
            (ew,  ["EW_thru",  "EW_left",  "EW_right"]),
            (we,  ["WE_thru",  "WE_left",  "WE_right"]),
            (ns,  ["NS_thru",  "NS_left",  "NS_right"]),
            (sn,  ["SN_thru",  "SN_left",  "SN_right"]),
        ]
        ratios = [THRU_RATIO, LEFT_RATIO, RIGHT_RATIO]

        for flow, routes in combos:
            for route, ratio in zip(routes, ratios):
                n = veh_per_interval(flow * ratio, dt)
                # 在區間內均勻隨機發車
                depart_times = sorted(
                    rng.uniform(t_start, t_end, n).tolist()
                )
                for dep in depart_times:
                    lines.append(
                        f'    <vehicle id="veh{veh_id}" type="car"'
                        f' route="{route}" depart="{dep:.2f}"/>'
                    )
                    veh_id += 1

    lines.append('</routes>')

    out_path = os.path.join(OUT_DIR, f"taipei_{pattern_name}.rou.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[generate_routes] 生成 {veh_id} 輛車 → {out_path}")
    return out_path


def generate_all():
    """生成所有交通情境的 .rou.xml"""
    for pattern in DEMAND_PATTERNS:
        generate_routes(pattern_name=pattern, seed=42)


if __name__ == "__main__":
    generate_all()
