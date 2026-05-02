"""
使用 netconvert 將 nodes.xml + edges.xml + connections.xml + tls.xml
合併成 zhongxiao_dunnan.net.xml。

執行方式：
    python sumo-taipei/network/build_net.py
"""
import subprocess
import sys
import os
from pathlib import Path

NET_DIR  = Path(__file__).resolve().parent
OUT_FILE = NET_DIR / "zhongxiao_dunnan.net.xml"


def find_netconvert():
    """尋找 netconvert 執行檔（支援 pip 安裝的 eclipse-sumo）"""
    # 1. 系統 PATH
    for candidate in ["netconvert", "netconvert.exe"]:
        result = subprocess.run(
            ["where" if sys.platform == "win32" else "which", candidate],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return candidate

    # 2. eclipse-sumo pip 套件
    try:
        import sumo
        sumo_home = Path(sumo.__file__).parent
        for candidate in [
            sumo_home / "bin" / "netconvert.exe",
            sumo_home / "bin" / "netconvert",
        ]:
            if candidate.exists():
                return str(candidate)
    except ImportError:
        pass

    # 3. SUMO_HOME 環境變數
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        for candidate in [
            Path(sumo_home) / "bin" / "netconvert.exe",
            Path(sumo_home) / "bin" / "netconvert",
        ]:
            if candidate.exists():
                return str(candidate)

    return None


def build():
    nc = find_netconvert()
    if nc is None:
        print("[ERROR] 找不到 netconvert，請確認 SUMO 已安裝或 SUMO_HOME 已設定。")
        sys.exit(1)

    cmd = [
        nc,
        "--node-files",       str(NET_DIR / "nodes.xml"),
        "--edge-files",       str(NET_DIR / "edges.xml"),
        "--connection-files", str(NET_DIR / "connections.xml"),
        "--tllogic-files",    str(NET_DIR / "tls.xml"),
        "--output-file",      str(OUT_FILE),
        "--no-internal-links", "false",
        "--junctions.join",   "false",
        "--tls.default-type", "static",
        "--geometry.remove",
        "--roundabouts.guess",
    ]

    print("[build_net] 執行:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[STDERR]", result.stderr)
        sys.exit(1)

    print(f"[build_net] 成功生成 → {OUT_FILE}")
    if result.stdout:
        print(result.stdout[:500])


if __name__ == "__main__":
    build()
