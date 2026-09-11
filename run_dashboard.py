"""
run_dashboard.py
Launcher for the CyberWorld SOC console (local SPAN or Containerlab Lab Mode).

  python run_dashboard.py --site local-default --interface eth1
  python run_dashboard.py --site containerlab-enterprise

Env:
  CYBERWORLD_SITE, CYBERWORLD_SITE_CONFIG, CYBERWORLD_SENSOR_IFACE
"""

import argparse
import os
import sys
from pathlib import Path
import threading
import time
import webbrowser

REPO_ROOT = Path(__file__).resolve().parent
bita_path = str(REPO_ROOT / "bita")

os.chdir(str(REPO_ROOT))
clean_path = [str(REPO_ROOT), bita_path]
for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
    if p and os.path.abspath(p) not in {os.path.abspath(x) for x in clean_path}:
        if os.path.isfile(os.path.join(p, "model.py")):
            continue
        clean_path.append(p)
os.environ["PYTHONPATH"] = os.pathsep.join(clean_path)
sys.path[:0] = [str(REPO_ROOT), bita_path]
for k in list(sys.modules):
    if k == "model" or k.startswith("model."):
        del sys.modules[k]


def open_browser(port: int):
    time.sleep(1.8)
    url = f"http://localhost:{port}"
    print(f"\n[+] Opening SOC Console: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[!] Could not auto-open browser: {e}. Open {url} manually.")


def _newest_mtime(root: Path) -> float:
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".git")]
        for name in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, name)))
            except OSError:
                pass
    return newest


def _ensure_frontend_build(repo_root: Path) -> None:
    """Build web_dashboard/dist when it is missing or older than src/.

    The backend serves the built bundle, so a stale dist silently shows an old UI
    even after the source changes — which looks exactly like a broken dashboard.
    """
    import shutil
    import subprocess

    web = repo_root / "web_dashboard"
    dist_index = web / "dist" / "index.html"
    src_newest = max(
        _newest_mtime(web / "src"),
        os.path.getmtime(web / "index.html") if (web / "index.html").exists() else 0.0,
        os.path.getmtime(web / "package.json") if (web / "package.json").exists() else 0.0,
    )
    dist_time = os.path.getmtime(dist_index) if dist_index.exists() else 0.0
    stale = dist_time < src_newest

    if dist_index.exists() and not stale:
        return

    reason = "not found" if not dist_index.exists() else "older than web_dashboard/src"
    print(f"[!] Frontend bundle {reason} — rebuilding with Vite...")
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        print("[!] npm is not on PATH. Build the dashboard manually:")
        print("      cd web_dashboard && npm install && npm run build")
        if not dist_index.exists():
            print("[!] No bundle at all — the SOC console will not render until you do.")
        else:
            print("[!] Serving the EXISTING (stale) bundle for now.")
        return
    if not (web / "node_modules").exists():
        print("[*] Installing dashboard dependencies (npm install)...")
        subprocess.run([npm, "install"], cwd=str(web), check=False)
    res = subprocess.run([npm, "run", "build"], cwd=str(web), check=False)
    if res.returncode != 0:
        print("[!] Vite build failed — see the output above. Serving whatever dist/ exists.")
    else:
        print("[+] Frontend bundle rebuilt.")


def _parse_args():
    p = argparse.ArgumentParser(description="CyberWorld SOC dashboard")
    p.add_argument("--site", default=None, help="Site id under config/sites/")
    p.add_argument("--site-config", default=None, help="Path to a site YAML file")
    p.add_argument("--interface", default=None, help="Override SPAN capture interface")
    p.add_argument(
        "--replay",
        default=None,
        metavar="PCAP",
        help="Demo mode: replay a PCAP capture instead of sniffing a NIC "
             "(e.g. --replay captures/live.pcap). No root / mirror NIC required.",
    )
    p.add_argument("--replay-speed", default=None, help="Replay speed multiplier (default 1.0)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.site_config:
        os.environ["CYBERWORLD_SITE_CONFIG"] = args.site_config
    if args.site:
        os.environ["CYBERWORLD_SITE"] = args.site
    if args.interface:
        os.environ["CYBERWORLD_SENSOR_IFACE"] = args.interface
    if args.replay:
        replay_path = os.path.abspath(args.replay)
        if not os.path.exists(replay_path):
            sys.exit(f"[!] Replay capture not found: {replay_path}")
        os.environ["CYBERWORLD_REPLAY_PCAP"] = replay_path
    if args.replay_speed:
        os.environ["CYBERWORLD_REPLAY_SPEED"] = str(args.replay_speed)

    from control_backend.site_config import get_site_config, reload_site_config

    reload_site_config()
    site = get_site_config()

    print("=" * 70)
    print("  CYBERWORLD SOC — SPAN DISCOVERY + DUAL-BRANCH / DEEPOP")
    print("=" * 70)
    print(f"[+] Repo root:     {REPO_ROOT}")
    print(f"[+] Site:          {site.site_id} (lab_mode={site.lab_mode})")
    print(f"[+] Sensor iface:  {site.sensor_interface}")
    print(f"[+] API & UI:      http://localhost:{args.port}")
    if os.environ.get("CYBERWORLD_REPLAY_PCAP"):
        print(f"[+] Mode:          PCAP REPLAY — {os.environ['CYBERWORLD_REPLAY_PCAP']}")
        print("[+] No mirror NIC or root required; SENSOR button replays the capture.")
    elif site.lab_mode:
        print("[+] Mode:          Lab Mode (Containerlab orchestration enabled)")
    else:
        print("[+] Mode:          Local SPAN (no Containerlab required)")
        print("[!] Capture needs CAP_NET_RAW / root on the mirror NIC.")
    print("=" * 70)

    _ensure_frontend_build(REPO_ROOT)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    import uvicorn

    uvicorn.run(
        "control_backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
