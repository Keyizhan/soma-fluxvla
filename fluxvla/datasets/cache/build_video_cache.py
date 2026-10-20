"""Build 8-frame Zstd caches with bounded RAM, atomic publication and resume."""

import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import uuid
from pathlib import Path

import numpy as np
from zstandard import ZstdError


BLOCK_FRAMES = 8


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _valid(entry, root, signature, settings):
    if entry.get("source") != signature or entry.get("settings") != settings:
        return False
    try:
        from .video_cache import VideoCache

        # Validate index geometry and both ends by decoding real blocks.
        cache = VideoCache.__new__(VideoCache)
        from collections import OrderedDict

        cache.directory = root
        cache._arrays = OrderedDict()
        cache._pid = os.getpid()
        cache.max_open_files = 1
        try:
            frames, pts, offsets = cache._open(entry)
            import zstandard

            decoder = zstandard.ZstdDecompressor()
            for block in {0, len(offsets) - 1}:
                start, size = map(int, offsets[block])
                n = min(8, entry["n"] - block * 8)
                expected = n * settings["height"] * settings["width"] * 3
                raw = decoder.decompress(
                    frames[start : start + size].tobytes(), max_output_size=expected
                )
                if len(raw) != expected:
                    return False
            return True
        finally:
            cache.close()
    except (OSError, ValueError, KeyError, RuntimeError, ZstdError):
        return False


def _build_one(job):
    import cv2
    import decord
    import zstandard

    source, relative, output, height, width, chunk_frames, level = job
    source, output = Path(source), Path(output)
    stat = source.stat()
    signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    settings = {
        "height": height,
        "width": width,
        "resize": "opencv-linear",
        "dtype": "uint8",
        "codec": "zstd",
        "block_frames": BLOCK_FRAMES,
        "compression_level": level,
        "decoder": "decord2-" + decord.__version__,
        "opencv": cv2.__version__,
    }
    identity = hashlib.sha256(
        json.dumps([relative, signature, settings], sort_keys=True).encode()
    ).hexdigest()
    record = output / "records" / (identity + ".json")
    if record.exists():
        try:
            entry = json.loads(record.read_text())
            if _valid(entry, output, signature, settings):
                return relative, entry, True
        except (OSError, ValueError):
            pass
    cv2.setNumThreads(1)
    reader = decord.VideoReader(str(source), ctx=decord.cpu(0), num_threads=1)
    n = len(reader)
    if n <= 0:
        raise ValueError(f"Video has no frames: {source}")
    target = output / "frames" / (identity + ".zstd")
    pts_target = output / "frames" / (identity + ".pts.npy")
    offsets_target = output / "frames" / (identity + ".offsets.npy")
    paths = (target, pts_target, offsets_target)
    tag = uuid.uuid4().hex
    temps = [p.with_name(p.name + "." + tag + ".tmp") for p in paths]
    pts = np.empty(n, dtype=np.float64)
    offsets = []
    encoder = zstandard.ZstdCompressor(level=level, threads=0, write_checksum=True)
    try:
        with temps[0].open("wb") as stream:
            # Consume complete 8-frame blocks. chunk_frames bounds decode batch size.
            batch_size = max(8, chunk_frames // 8 * 8)
            for start in range(0, n, batch_size):
                indices = list(range(start, min(start + batch_size, n)))
                # Sequential reads match the fast cloud build path. Some codecs
                # have inconsistent get_batch buffer sizes in Decord2.
                decoded = np.stack([reader[index].asnumpy() for index in indices])
                if decoded.dtype != np.uint8 or decoded.ndim != 4 or decoded.shape[-1] != 3:
                    raise ValueError(
                        f"Decoder did not return uint8 RGB for {source}: "
                        f"shape={decoded.shape}, dtype={decoded.dtype}"
                    )
                pts[start : start + len(indices)] = np.asarray(reader.get_frame_timestamp(indices))[
                    :, 0
                ]
                for i in range(0, len(indices), 8):
                    block = decoded[i : i + 8]
                    if block.shape[1:3] != (height, width):
                        block = np.stack(
                            [
                                cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
                                for f in block
                            ]
                        )
                    raw = block.tobytes()
                    compressed = encoder.compress(raw)
                    offsets.append((stream.tell(), len(compressed)))
                    stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
        if not np.isfinite(pts).all() or np.any(np.diff(pts) < 0):
            raise ValueError(f"Invalid source timestamps: {source}")
        if signature != {"size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}:
            raise ValueError(f"Source changed during decoding: {source}")
        for path, array in ((temps[1], pts), (temps[2], np.asarray(offsets, dtype=np.int64))):
            with path.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        for temp, path in zip(temps, paths):
            os.replace(temp, path)
        entry = {
            "cache": target.relative_to(output).as_posix(),
            "codec": "zstd",
            "block_frames": 8,
            "timestamps": pts_target.relative_to(output).as_posix(),
            "offsets": offsets_target.relative_to(output).as_posix(),
            "n": n,
            "fps": float(reader.get_avg_fps()),
            "source": signature,
            "settings": settings,
        }
        atomic_json(record, entry)
        return relative, entry, False
    finally:
        for path in temps:
            path.unlink(missing_ok=True)


def build_cache(
    data_root, directory, height=224, width=224, workers=4, chunk_frames=16, compression_level=3
):
    if min(height, width, workers, chunk_frames) < 1:
        raise ValueError("Dimensions, workers and chunk_frames must be positive")
    data_root, directory = Path(data_root).resolve(), Path(directory).resolve()
    videos = sorted((data_root / "videos").rglob("*.mp4"))
    if not videos:
        raise ValueError(f"No videos found under {data_root / 'videos'}")
    for part in ("frames", "records"):
        (directory / part).mkdir(parents=True, exist_ok=True)
    with (directory / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = {}
        jobs = [
            (
                str(p),
                p.relative_to(data_root).as_posix(),
                str(directory),
                height,
                width,
                chunk_frames,
                compression_level,
            )
            for p in videos
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for key, entry, skipped in pool.map(_build_one, jobs):
                entries[key] = entry
                print(
                    f"[{len(entries)}/{len(videos)}] {'reuse' if skipped else 'built'} {key}",
                    flush=True,
                )
        atomic_json(directory / "index.json", {"version": 2, "entries": entries})
    return entries
