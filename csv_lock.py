# pylint: skip-file
from __future__ import annotations
import os, time, random, pathlib
import pandas as pd

def _acquire_lock(lock_dir: pathlib.Path, timeout: float = 600, poll: float = 0.2) -> None:
    lock_dir = pathlib.Path(lock_dir)
    start = time.time()
    while True:
        try:
            lock_dir.mkdir(exist_ok=False)  # atomic on POSIX/NFS
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for lock {lock_dir}")
            time.sleep(poll + random.random() * 0.3)

def _release_lock(lock_dir: pathlib.Path) -> None:
    try:
        pathlib.Path(lock_dir).rmdir()
    except FileNotFoundError:
        pass

def update_csv_row(csv_path: str | os.PathLike, row_idx: int, updates: dict) -> None:
    """
    Atomically update selected columns in a single CSV row.
    - csv_path: path to shared CSV
    - row_idx: zero-based row index
    - updates: mapping of column -> value
    """
    csv_path = pathlib.Path(csv_path)
    lock_dir = csv_path.with_suffix(csv_path.suffix + ".lockdir")

    _acquire_lock(lock_dir)
    try:
        df = pd.read_csv(csv_path)
        if row_idx < 0 or row_idx >= len(df):
            raise IndexError(f"Row {row_idx} out of range (len={len(df)}) for {csv_path}")
        # add any missing columns (defensive)
        for col in updates.keys():
            if col not in df.columns:
                df[col] = 0
        for k, v in updates.items():
            df.at[row_idx, k] = v
        tmp = csv_path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, csv_path)  # atomic replace
    finally:
        _release_lock(lock_dir)