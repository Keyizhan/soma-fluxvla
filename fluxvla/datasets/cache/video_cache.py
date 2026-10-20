"""Read 8-frame Zstd RGB blocks shared by all ranks via OS page cache.

New caches use dataset-relative keys and actual frame timestamps. Legacy cloud
index.json files (absolute keys, fps/n metadata) are also readable.
"""

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np


class VideoCache:
    def __init__(self, directory, data_root=None, max_open_files=64):
        self.directory = Path(directory).resolve()
        self.data_root = Path(data_root).resolve() if data_root else None
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        with (self.directory / "index.json").open() as stream:
            index = json.load(stream)
        self.version = index.get("version", 1)
        if self.version not in (1, 2):
            raise ValueError(f"Unsupported cache index version: {self.version}")
        self.entries = index["entries"] if self.version == 2 else index
        self._arrays = OrderedDict()
        self._pid = os.getpid()

    def _entry(self, video_path):
        path = Path(video_path).resolve()
        if self.version == 2:
            if self.data_root is None:
                raise ValueError("data_root is required for a portable v2 cache")
            key = path.relative_to(self.data_root).as_posix()
        else:
            key = str(video_path)
            if key not in self.entries:
                key = str(path)
        try:
            entry = self.entries[key]
        except KeyError as exc:
            raise KeyError(f"Video is absent from cache: {video_path}. Rebuild the cache.") from exc
        return path, entry

    def _path(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        path = (self.directory / path).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Cache path escapes cache directory")
        return path

    def _open(self, entry):
        if os.getpid() != self._pid:
            self.close()
            self._pid = os.getpid()
        key = entry["cache"]
        if key not in self._arrays:
            offsets = None
            if entry.get("codec") == "zstd":
                if entry.get("block_frames") != 8:
                    raise ValueError("Only 8-frame Zstd blocks are supported")
                frames = np.memmap(self._path(key), mode="r", dtype=np.uint8)
                offsets = np.load(self._path(entry["offsets"]), mmap_mode="r", allow_pickle=False)
                blocks = (entry["n"] + 7) // 8
                if (
                    offsets.shape != (blocks, 2)
                    or not blocks
                    or not np.issubdtype(offsets.dtype, np.integer)
                    or np.any(offsets[:, 1] <= 0)
                    or np.any(offsets[:, 0] < 0)
                    or np.any(offsets[:, 0] + offsets[:, 1] > len(frames))
                ):
                    raise ValueError(f"Invalid block offsets: {key}")
                if offsets[0, 0] != 0 or np.any(offsets[1:, 0] != offsets[:-1].sum(axis=1)):
                    raise ValueError(f"Noncontiguous block offsets: {key}")
                if int(offsets[-1].sum()) != len(frames):
                    raise ValueError(f"Compressed file size mismatch: {key}")
            else:
                frames = np.load(self._path(key), mmap_mode="r", allow_pickle=False)
                if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                    raise ValueError(f"Invalid cached RGB array: {key}")
                if len(frames) != entry["n"] or not len(frames):
                    raise ValueError(f"Cached frame count mismatch: {key}")
            pts = None
            if "timestamps" in entry:
                pts = np.load(self._path(entry["timestamps"]), mmap_mode="r", allow_pickle=False)
                if (
                    pts.shape != (entry["n"],)
                    or not np.isfinite(pts).all()
                    or np.any(np.diff(pts) < 0)
                ):
                    raise ValueError(f"Invalid timestamps: {key}")
            elif not np.isfinite(entry.get("fps", 0)) or entry.get("fps", 0) <= 0:
                raise ValueError(f"Missing/invalid fps in legacy cache: {key}")
            self._arrays[key] = frames, pts, offsets
        self._arrays.move_to_end(key)
        while len(self._arrays) > self.max_open_files:
            _, arrays = self._arrays.popitem(last=False)
            for array in arrays:
                if array is not None:
                    array._mmap.close()
        return self._arrays[key]

    def read(self, video_path, timestamps, tolerance_s=0.1):
        """Return a private contiguous [T, 3, H, W] uint8 array.

        Misses and stale entries fail explicitly; no expensive hidden MP4 fallback.
        """
        query = np.asarray(timestamps, dtype=np.float64)
        if query.ndim != 1 or not query.size or not np.isfinite(query).all():
            raise ValueError("timestamps must be a nonempty finite sequence")
        if not np.isfinite(tolerance_s) or tolerance_s < 0:
            raise ValueError("tolerance_s must be finite and nonnegative")
        source, entry = self._entry(video_path)
        if "source" in entry:
            stat = source.stat()
            if entry["source"] != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                raise ValueError(f"Stale cache for {source}; rebuild it")
        frames, pts, offsets = self._open(entry)
        if pts is None:
            indices = np.clip(np.rint(query * entry["fps"]), 0, entry["n"] - 1).astype(np.int64)
            distances = np.abs(indices / entry["fps"] - query)
        else:
            right = np.clip(np.searchsorted(pts, query), 0, len(pts) - 1)
            left = np.clip(right - 1, 0, len(pts) - 1)
            indices = np.where(np.abs(pts[left] - query) <= np.abs(pts[right] - query), left, right)
            distances = np.abs(pts[indices] - query)
        if np.any(distances > tolerance_s):
            raise ValueError(f"Cached timestamp outside tolerance for {source}: {distances}")
        if offsets is None:
            selected = frames[indices]
        else:
            import zstandard

            decoder = zstandard.ZstdDecompressor()
            height, width = entry["settings"]["height"], entry["settings"]["width"]
            selected = np.empty((len(indices), height, width, 3), dtype=np.uint8)
            for block in np.unique(indices // 8):
                start, size = map(int, offsets[block])
                count = min(8, entry["n"] - int(block) * 8)
                expected = count * height * width * 3
                raw = decoder.decompress(
                    frames[start : start + size].tobytes(), max_output_size=expected
                )
                if len(raw) != expected:
                    raise ValueError(f"Decoded block size mismatch: {entry['cache']}")
                decoded = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)
                mask = indices // 8 == block
                selected[mask] = decoded[indices[mask] % 8]
        return np.ascontiguousarray(selected.transpose(0, 3, 1, 2))

    def close(self):
        for arrays in self._arrays.values():
            for array in arrays:
                if array is not None:
                    array._mmap.close()
        self._arrays.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        return state
