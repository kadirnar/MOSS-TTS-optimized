import os
import io
import json
import fcntl
import errno
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch


DEFAULT_META_FILENAME = "shards.meta.json"
DEFAULT_BIN_PATTERN = "data-{shard_id:05d}.bin"
DEFAULT_IDX_PATTERN = "data-{shard_id:05d}.idx"
DEFAULT_IDX_BIN_PATTERN = "data-{shard_id:05d}.idx.bin"


def _get_rank_and_world_size() -> Tuple[int, int]:
    try:
        # Accelerate / DDP envs
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        rank, world_size = 0, 1
    return rank, world_size


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class FileLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        _ensure_dir(os.path.dirname(lock_path))
        # Create lock file if not exists
        self.fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)

    def acquire(self):
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX)
                break
            except OSError as e:
                if e.errno != errno.EINTR:
                    raise

    def release(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


class ShardedBinIdxWriter:
    """
    Multi-shard cache writer. Each sample is serialized as a single NPZ blob and
    appended to a shard's .bin file, with an entry in the shard's .idx text file:
    "data_id\toffset\tlength\n".

    Concurrency safety:
    - Use a per-shard lock file to serialize append operations across processes.
    - Meta file creation is idempotent and verified for consistency.

    Monotonicity:
    - Enforces per-shard strictly increasing data_id on append. If violated, raises ValueError.
    """

    def __init__(self, cache_dir: str, num_shards: int = 16, meta_filename: str = DEFAULT_META_FILENAME):
        self.cache_dir = cache_dir
        self.num_shards = int(num_shards)
        self.meta_filename = meta_filename
        _ensure_dir(self.cache_dir)

        self.meta_path = os.path.join(self.cache_dir, self.meta_filename)
        self._init_or_validate_meta()

    def _init_or_validate_meta(self):
        rank, _ = _get_rank_and_world_size()
        if not os.path.exists(self.meta_path):
            # Create atomically using a lock on the meta file path
            lock = FileLock(self.meta_path + ".lock")
            with lock:
                if not os.path.exists(self.meta_path):
                    meta = {
                        "version": 1,
                        "record_format": "npz",
                        "num_shards": self.num_shards,
                        "bin_pattern": DEFAULT_BIN_PATTERN,
                        "idx_pattern": DEFAULT_IDX_PATTERN,
                    }
                    with open(self.meta_path, "w") as f:
                        json.dump(meta, f)
        # Validate
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        if meta.get("num_shards") != self.num_shards:
            raise ValueError(f"num_shards mismatch: existing {meta.get('num_shards')} vs new {self.num_shards}")

    def _shard_paths(self, shard_id: int) -> Tuple[str, str, str, str]:
        bin_path = os.path.join(self.cache_dir, DEFAULT_BIN_PATTERN.format(shard_id=shard_id))
        idx_path = os.path.join(self.cache_dir, DEFAULT_IDX_PATTERN.format(shard_id=shard_id))
        idx_bin_path = os.path.join(self.cache_dir, DEFAULT_IDX_BIN_PATTERN.format(shard_id=shard_id))
        lock_path = idx_path + ".lock"
        return bin_path, idx_path, idx_bin_path, lock_path

    @staticmethod
    def _to_numpy_inputs(sample: Dict[str, Any]) -> Dict[str, Any]:
        numpy_inputs: Dict[str, Any] = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.bfloat16:
                    numpy_inputs[key] = value.detach().cpu().half().numpy()
                else:
                    numpy_inputs[key] = value.detach().cpu().numpy()
            else:
                numpy_inputs[key] = value
        return numpy_inputs

    def write_sample(self, data_id: int, sample: Dict[str, Any], compress: bool = True):
        shard_id = int(data_id) % self.num_shards
        bin_path, idx_path, idx_bin_path, lock_path = self._shard_paths(shard_id)
        _ensure_dir(os.path.dirname(bin_path))
        _ensure_dir(os.path.dirname(idx_path))

        numpy_inputs = self._to_numpy_inputs(sample)

        # Serialize to NPZ bytes first
        buffer = io.BytesIO()
        # Use compressed to reduce disk
        if compress:
            np.savez_compressed(buffer, **numpy_inputs)
        else:
            np.savez(buffer, **numpy_inputs)
        blob = buffer.getvalue()

        # Serialize append operations under a single lock
        lock = FileLock(lock_path)
        with lock:
            # Enforce per-shard strictly increasing data_id by checking the last record
            last_id = None
            if os.path.exists(idx_bin_path):
                file_size = os.path.getsize(idx_bin_path)
                if file_size % (8 * 3) != 0:
                    raise ValueError(f"Corrupted idx.bin file: {idx_bin_path}")
                if file_size >= (8 * 3):
                    with open(idx_bin_path, "rb") as bif:
                        bif.seek(file_size - (8 * 3))
                        tail = np.fromfile(bif, dtype=np.int64, count=3)
                        if tail.size == 3:
                            last_id = int(tail[0])
            if last_id is not None and int(data_id) <= last_id:
                raise ValueError(
                    f"data_id for shard {shard_id} must be strictly increasing; last={last_id}, got={int(data_id)}"
                )
            # Append NPZ bytes to the .bin shard.
            with open(bin_path, "ab") as bf:
                offset = bf.tell()
                bf.write(blob)
                length = len(blob)
            # Append textual idx for human readability/debug
            with open(idx_path, "a") as inf:
                inf.write(f"{int(data_id)}\t{int(offset)}\t{int(length)}\n")
            # Append binary idx for O(1) random access at read time.
            with open(idx_bin_path, "ab") as bif:
                np.array([int(data_id), int(offset), int(length)], dtype=np.int64).tofile(bif)


class ShardedBinIdxReader:
    """
    Reader for multi-shard bin+idx cache written by ShardedBinIdxWriter.

    Loads per-shard index into memory on first access for O(1) lookup.
    """

    def __init__(self, cache_dir: str, meta_filename: str = DEFAULT_META_FILENAME):
        self.cache_dir = cache_dir
        self.meta_path = os.path.join(cache_dir, meta_filename)
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        self.num_shards = int(meta["num_shards"])
        self.record_format = meta.get("record_format", "npz")
        if self.record_format != "npz":
            raise ValueError(f"Unsupported record_format: {self.record_format}")
        # Cached per-shard structures: (mm, sort_idx, sorted_keys)
        self._shard_arrays: Dict[int, Tuple[np.memmap, np.ndarray, np.ndarray]] = {}

    def _shard_paths(self, shard_id: int) -> Tuple[str, str, str]:
        bin_path = os.path.join(self.cache_dir, DEFAULT_BIN_PATTERN.format(shard_id=shard_id))
        idx_path = os.path.join(self.cache_dir, DEFAULT_IDX_PATTERN.format(shard_id=shard_id))
        idx_bin_path = os.path.join(self.cache_dir, DEFAULT_IDX_BIN_PATTERN.format(shard_id=shard_id))
        return bin_path, idx_path, idx_bin_path

    def _ensure_binary_idx(self, idx_path: str, idx_bin_path: str):
        if os.path.exists(idx_bin_path):
            return
        if not os.path.exists(idx_path):
            # Nothing to build
            return
        # Build binary idx from text idx (one-time cost)
        rows: List[List[int]] = []
        with open(idx_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                try:
                    did = int(parts[0])
                    off = int(parts[1])
                    leng = int(parts[2])
                    rows.append([did, off, leng])
                except Exception:
                    continue
        if len(rows) == 0:
            return
        arr = np.asarray(rows, dtype=np.int64)
        with open(idx_bin_path, "wb") as bif:
            arr.tofile(bif)

    def _load_idx_if_needed(self, shard_id: int):
        if shard_id in self._shard_arrays:
            return
        _, idx_path, idx_bin_path = self._shard_paths(shard_id)
        self._ensure_binary_idx(idx_path, idx_bin_path)
        if not os.path.exists(idx_bin_path):
            # Empty shard
            self._shard_arrays[shard_id] = (np.memmap(idx_bin_path, mode='w+', dtype=np.int64, shape=(0,)), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
            return
        file_size = os.path.getsize(idx_bin_path)
        if file_size == 0:
            self._shard_arrays[shard_id] = (np.memmap(idx_bin_path, mode='r', dtype=np.int64, shape=(0,)), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
            return
        if file_size % (8 * 3) != 0:
            raise ValueError(f"Corrupted idx.bin file: {idx_bin_path}")
        num_rows = file_size // (8 * 3)
        mm = np.memmap(idx_bin_path, mode='r', dtype=np.int64, shape=(num_rows * 3,))
        mm = mm.reshape(num_rows, 3)
        keys = mm[:, 0]
        # Use stable sort; build sorted view
        sort_idx = np.argsort(keys, kind='mergesort')
        sorted_keys = keys[sort_idx]
        self._shard_arrays[shard_id] = (mm, sort_idx, sorted_keys)

    def has(self, data_id: int) -> bool:
        shard_id = int(data_id) % self.num_shards
        self._load_idx_if_needed(shard_id)
        mm, sort_idx, sorted_keys = self._shard_arrays.get(shard_id, (None, None, None))
        if mm is None or sorted_keys is None or len(sorted_keys) == 0:
            return False
        did = int(data_id)
        pos = np.searchsorted(sorted_keys, did, side='right') - 1
        if pos < 0 or pos >= len(sorted_keys):
            return False
        return sorted_keys[pos] == did

    def get(self, data_id: int) -> Optional[Dict[str, Any]]:
        shard_id = int(data_id) % self.num_shards
        bin_path, _, _ = self._shard_paths(shard_id)
        self._load_idx_if_needed(shard_id)
        mm, sort_idx, sorted_keys = self._shard_arrays.get(shard_id, (None, None, None))
        if mm is None or len(sorted_keys) == 0:
            return None
        did = int(data_id)
        pos = np.searchsorted(sorted_keys, did, side='right') - 1
        if pos < 0 or pos >= len(sorted_keys) or sorted_keys[pos] != did:
            return None
        row = int(sort_idx[pos])
        offset = int(mm[row, 1])
        length = int(mm[row, 2])
        if not os.path.exists(bin_path):
            return None
        with open(bin_path, "rb") as bf:
            bf.seek(offset)
            blob = bf.read(length)
        buffer = io.BytesIO(blob)
        npz = np.load(buffer)
        loaded: Dict[str, Any] = {}
        for k, v in npz.items():
            if hasattr(v, "dtype"):
                loaded[k] = torch.from_numpy(np.array(v))
            else:
                loaded[k] = v
        return loaded


