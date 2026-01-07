# pylint: skip-file
from __future__ import annotations
import os, time, random, pathlib, socket, tempfile
import pandas as pd
from typing import Mapping

_LOCK_POLL = 0.2
_LOCK_JITTER = 0.3
_LOCK_TIMEOUT = 600         # seconds to wait before giving up
_LOCK_STALE_AFTER = 3600    # seconds before we consider breaking a stale lock

def _lock_paths(csv_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    lockdir = csv_path.with_suffix(csv_path.suffix + ".lockdir")
    meta = lockdir / "owner.txt"
    return lockdir, meta

def _acquire_lock(csv_path: pathlib.Path,
                  timeout: float = _LOCK_TIMEOUT,
                  stale_after: float = _LOCK_STALE_AFTER) -> pathlib.Path:
    lockdir, meta = _lock_paths(csv_path)
    start = time.time()
    while True:
        try:
            lockdir.mkdir(exist_ok=False)  # POSIX-atomic even on NFS (usually)
            meta.write_text(f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}\n",
                            encoding="utf-8")
            return lockdir
        except FileExistsError:
            # detect stale lock
            try:
                mtime = lockdir.stat().st_mtime
            except FileNotFoundError:
                continue
            if time.time() - mtime > stale_after:
                # best-effort stale lock break
                try:
                    # remove marker then dir
                    try:
                        meta.unlink(missing_ok=True)
                    except TypeError:  # py<3.8
                        try: meta.unlink()
                        except FileNotFoundError: pass
                    lockdir.rmdir()
                    # retry loop will acquire on next iteration
                    continue
                except OSError:
                    pass
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for lock {lockdir}")
            time.sleep(_LOCK_POLL + random.random() * _LOCK_JITTER)

def _release_lock(lockdir: pathlib.Path) -> None:
    try:
        # cleanup marker if present
        owner = lockdir / "owner.txt"
        try:
            owner.unlink()
        except FileNotFoundError:
            pass
        lockdir.rmdir()
    except FileNotFoundError:
        pass

def update_csv_row(csv_path: str | os.PathLike,
                   row_idx: int,
                   updates: Mapping[str, object]) -> None:
    """
    Atomically update selected columns in a single CSV row.
    Safe for concurrent writers on Linux/NFS (best-effort).
    """
    csv_path = pathlib.Path(csv_path)
    lockdir = _acquire_lock(csv_path)
    try:
        # Read -> modify -> write to temp -> fsync -> atomic replace
        df = pd.read_csv(csv_path)
        if not (0 <= row_idx < len(df)):
            raise IndexError(f"Row {row_idx} out of range (len={len(df)}) for {csv_path}")
        for col in updates.keys():
            if col not in df.columns:
                df[col] = pd.NA
        for k, v in updates.items():
            df.at[row_idx, k] = v

        # write atomically to same directory
        fd, tmp_path = tempfile.mkstemp(dir=csv_path.parent,
                                        prefix=csv_path.name + ".tmp.",
                                        text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                df.to_csv(fh, index=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, csv_path)  # atomic on same filesystem
        finally:
            # if anything failed before replace, clean up
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
    finally:
        _release_lock(lockdir)