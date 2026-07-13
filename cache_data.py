"""One-shot script: load all parquets and save a single feather cache.

Run once; downstream scripts then load from `data_cache.pkl` in <2 s.
"""
from __future__ import annotations
import pickle
import time
from pathlib import Path

from data import load_macro, load_panel, load_sector_map

CACHE = Path(__file__).parent / "data_cache.pkl"


def main() -> None:
    t0 = time.time()
    panel = load_panel()
    macro = load_macro()
    sm = load_sector_map()
    print(f"loaded in {time.time()-t0:.1f}s")
    print(f"  close panel: {panel['close'].shape}")
    print(f"  macro: {macro.shape}")
    print(f"  sector entries: {len(sm)}")
    with open(CACHE, "wb") as f:
        pickle.dump({"panel": panel, "macro": macro, "sector_map": sm}, f)
    print(f"saved cache → {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
