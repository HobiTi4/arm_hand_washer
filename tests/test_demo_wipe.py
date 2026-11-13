"""Регресійний тест: демо притиску тримає ~20 Н без втрати контакту."""
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "sim" / "xarm7_window_head" / "scripts"


def run_demo(*args):
    return subprocess.run([sys.executable, "demo_wipe.py", *args], cwd=SCRIPTS, capture_output=True, text=True, check=True).stdout


def test_compliant_holder_keeps_force_and_contact():
    out = run_demo()
    mean = float(re.search(r"mean ([\d.]+) N", out).group(1))
    lost = float(re.search(r"contact lost in stroke: ([\d.]+)%", out).group(1))
    assert 19.0 <= mean <= 21.0
    assert lost == 0.0
    assert "frame/sill contacts: 0 samples" in out


def test_rigid_holder_loses_contact():
    out = run_demo("--rigid")
    lost = float(re.search(r"contact lost in stroke: ([\d.]+)%", out).group(1))
    assert lost > 5.0
