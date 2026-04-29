"""
TDX 車流量爬蟲
忠孝東路 x 敦化南路 路口

USE_MOCK = True  → 使用模擬資料（不需帳號）
USE_MOCK = False → 使用真實 TDX API（需填入 CLIENT_ID / CLIENT_SECRET）
"""

import json
import requests

# ──────────────────────────────────────────
# 設定區
# ──────────────────────────────────────────
USE_MOCK = True           # 切換 Mock / 真實 TDX

CLIENT_ID     = "YOUR_CLIENT_ID"      # TDX 帳號 → 會員中心 → API金鑰
CLIENT_SECRET = "YOUR_CLIENT_SECRET"

TDX_AUTH_URL  = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
TDX_API_BASE  = "https://tdx.transportdata.tw/api/basic"

# 忠孝東路 x 敦化南路 附近的 VD 感測器關鍵字
TARGET_ROADS  = ["忠孝東路", "敦化南路"]

# ──────────────────────────────────────────
# Mock 資料（模擬三個時段）
# ──────────────────────────────────────────
MOCK_DATA = {
    "morning_peak": [   # 早尖峰 07:00–09:00
        {"direction": "EB", "volume_per_hour": 1200},
        {"direction": "WB", "volume_per_hour": 1000},
        {"direction": "NB", "volume_per_hour": 800},
        {"direction": "SB", "volume_per_hour": 700},
    ],
    "off_peak": [        # 離峰 09:00–17:00
        {"direction": "EB", "volume_per_hour": 550},
        {"direction": "WB", "volume_per_hour": 480},
        {"direction": "NB", "volume_per_hour": 380},
        {"direction": "SB", "volume_per_hour": 320},
    ],
    "evening_peak": [    # 晚尖峰 17:00–19:00
        {"direction": "EB", "volume_per_hour": 1400},
        {"direction": "WB", "volume_per_hour": 1300},
        {"direction": "NB", "volume_per_hour": 950},
        {"direction": "SB", "volume_per_hour": 880},
    ],
}


# ──────────────────────────────────────────
# TDX 認證
# ──────────────────────────────────────────
def get_tdx_token() -> str:
    resp = requests.post(TDX_AUTH_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


# ──────────────────────────────────────────
# TDX 真實資料
# ──────────────────────────────────────────
def fetch_tdx_traffic(token: str) -> list:
    """
    呼叫 TDX VD 感測器 API，過濾忠孝東路 / 敦化南路附近的車流資料。
    回傳與 mock 相同格式的 list。
    """
    url = f"{TDX_API_BASE}/v2/Road/Traffic/Live/VD/City/Taipei"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "$filter": " or ".join(
            [f"contains(RoadName,'{r}')" for r in TARGET_ROADS]
        ),
        "$format": "JSON",
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    raw = resp.json()

    results = []
    dir_map = {"E": "EB", "W": "WB", "N": "NB", "S": "SB"}
    for vd in raw.get("VDs", []):
        for link in vd.get("DetectionLinks", []):
            bearing = link.get("Bearing", "")
            if bearing in dir_map:
                # Lanes 資料裡有 Volume（輛/5min），轉換成輛/hr
                for lane in link.get("Lanes", []):
                    vol_5min = lane.get("Volume", 0)
                    results.append({
                        "direction":       dir_map[bearing],
                        "volume_per_hour": vol_5min * 12,
                        "vd_id":           vd.get("VDID", ""),
                        "road_name":       link.get("RoadName", ""),
                    })
    return results


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────
def get_traffic_data(period: str = "morning_peak") -> list:
    """
    回傳車流資料 list。

    period 選項（mock 模式）:
        'morning_peak' | 'off_peak' | 'evening_peak'

    真實 TDX 模式時 period 參數無作用（抓即時資料）。
    """
    if USE_MOCK:
        data = MOCK_DATA.get(period, MOCK_DATA["morning_peak"])
        for d in data:
            d["period"] = period
        return data
    else:
        token = get_tdx_token()
        return fetch_tdx_traffic(token)


# ──────────────────────────────────────────
if __name__ == "__main__":
    for period in ["morning_peak", "off_peak", "evening_peak"]:
        print(f"\n{'='*40}")
        print(f"時段：{period}")
        print('='*40)
        data = get_traffic_data(period)
        print(json.dumps(data, ensure_ascii=False, indent=2))
