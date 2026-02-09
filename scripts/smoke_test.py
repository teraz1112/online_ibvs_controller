from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable,
        "isvs.py",
        "--scenario",
        "dino",
        "--target",
        "data/samples/target.png",
        "--mask",
        "data/samples/mask.png",
        "--observed",
        "data/samples/observed.png",
        "--output-dir",
        str(out_dir),
        "--max-iter",
        "1",
        "--use-dummy-extractor",
    ]
    subprocess.run(cmd, cwd=root, check=True)

    expected = out_dir / "valdata.csv"
    if not expected.exists():
        raise SystemExit(f"Smoke test failed: missing {expected}")
    print(f"[OK] Smoke test passed: {expected}")


if __name__ == "__main__":
    main()
