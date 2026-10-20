"""Measure 8-frame Zstd and the actual training reader on a bounded video sample."""

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import zstandard

from fluxvla.datasets.cache import VideoCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--videos-per-camera", type=int, default=6)
    parser.add_argument("--reads", type=int, default=300)
    parser.add_argument(
        "--compare", action="store_true", help="Also benchmark other codecs/block sizes"
    )
    args = parser.parse_args()
    if min(args.videos_per_camera, args.reads) < 1:
        parser.error("Sample counts must be positive")
    cache = VideoCache(args.cache_dir, args.data_root)
    if cache.version == 2 and not args.data_root:
        parser.error("--data-root is required for v2 caches")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(417)
    by_camera = {}
    for key, entry in cache.entries.items():
        if entry["n"] >= 32:
            by_camera.setdefault(Path(key).parent.name, []).append((key, entry))
    chosen = [
        item
        for items in by_camera.values()
        for item in rng.sample(items, min(args.videos_per_camera, len(items)))
    ]
    if not chosen:
        parser.error("Benchmark requires at least one video with 32 frames")
    windows, sources = [], []
    full_raw = 0
    for entry in cache.entries.values():
        if "settings" in entry:
            height, width = entry["settings"]["height"], entry["settings"]["width"]
        else:
            array = np.load(cache._path(entry["cache"]), mmap_mode="r")
            height, width = array.shape[1:3]
            array._mmap.close()
        full_raw += entry["n"] * height * width * 3
    try:
        for key, entry in chosen:
            source = Path(args.data_root) / key if cache.version == 2 else Path(key)
            pts = (
                np.load(cache._path(entry["timestamps"]))
                if "timestamps" in entry
                else np.arange(entry["n"]) / entry["fps"]
            )
            sources.append((source, pts))
            for fraction in (0.1, 0.5, 0.85):
                start = min(int(len(pts) * fraction), len(pts) - 32)
                decoded = cache.read(source, pts[start : start + 32]).transpose(0, 2, 3, 1)
                windows.append((Path(key).parent.name, np.ascontiguousarray(decoded)))
        sample = {
            "videos": len(chosen),
            "frames": len(windows) * 32,
            "raw_MiB": sum(a.nbytes for _, a in windows) / 2**20,
            "conditions": "single process, warm file pages; no cold-storage or DDP scaling claim",
        }
        print(json.dumps(sample), flush=True)
        rows = []
        variants = [(8, "zstd3")]
        if args.compare:
            variants = [
                (n, codec) for n in (1, 8, 32) for codec in ("lz4", "zstd1", "zstd3", "delta_zstd1")
            ]
        for block_frames, codec in variants:
            if codec == "lz4":
                import lz4.frame
            encoder = zstandard.ZstdCompressor(
                level=3 if codec == "zstd3" else 1, threads=0, write_checksum=True
            )
            decoder = zstandard.ZstdDecompressor()

            def encode(array):
                if codec.startswith("delta"):
                    array = np.concatenate([array[:1], array[1:] - array[:-1]])
                data = array.tobytes()
                return lz4.frame.compress(data) if codec == "lz4" else encoder.compress(data)

            def decode(data, shape):
                raw = lz4.frame.decompress(data) if codec == "lz4" else decoder.decompress(data)
                array = np.frombuffer(raw, np.uint8).reshape(shape)
                return (
                    np.cumsum(array, axis=0, dtype=np.uint8) if codec.startswith("delta") else array
                )

            chunks = [
                (cam, np.ascontiguousarray(a[i : i + block_frames]))
                for cam, a in windows
                for i in range(0, 32, block_frames)
            ]
            blobs, offsets = [], []
            compress_time = decompress_time = 0.0
            raw_size = compressed_size = 0
            for _, array in chunks:
                before = time.perf_counter()
                blob = encode(array)
                compress_time += time.perf_counter() - before
                before = time.perf_counter()
                restored = decode(blob, array.shape)
                decompress_time += time.perf_counter() - before
                np.testing.assert_array_equal(restored, array)
                blobs.append(blob)
                raw_size += array.nbytes
                compressed_size += len(blob)
            latencies = []
            with tempfile.TemporaryFile(dir=output) as stream:
                for blob in blobs:
                    offsets.append((stream.tell(), len(blob)))
                    stream.write(blob)
                stream.flush()
                os.fsync(stream.fileno())
                for _ in range(args.reads):
                    index = rng.randrange(len(chunks))
                    offset, size = offsets[index]
                    before = time.perf_counter()
                    stream.seek(offset)
                    restored = decode(stream.read(size), chunks[index][1].shape)
                    restored[rng.randrange(block_frames)].copy()
                    latencies.append(time.perf_counter() - before)
            row = {
                "codec": codec,
                "block_frames": block_frames,
                "ratio": raw_size / compressed_size,
                "projected_4x_GiB": full_raw * 4 * compressed_size / raw_size / 2**30,
                "compression_MiB_s": raw_size / compress_time / 2**20,
                "decompression_MiB_s": raw_size / decompress_time / 2**20,
                "random_frame_fps_1core": len(latencies) / sum(latencies),
                "random_p95_ms": float(np.percentile(latencies, 95)) * 1000,
                "lossless": True,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        reader_times = []
        for _ in range(args.reads):
            source, pts = rng.choice(sources)
            timestamp = float(pts[rng.randrange(len(pts))])
            before = time.perf_counter()
            cache.read(source, [timestamp])
            reader_times.append(time.perf_counter() - before)
        reader = {
            "random_frame_fps_1core": len(reader_times) / sum(reader_times),
            "random_p95_ms": float(np.percentile(reader_times, 95)) * 1000,
            "conditions": sample["conditions"],
        }
        for name, value in [
            ("sample.json", sample),
            ("results.json", rows),
            ("reader_results.json", reader),
        ]:
            (output / name).write_text(json.dumps(value, indent=2) + "\n")
        print("training_reader", json.dumps(reader), flush=True)
    finally:
        cache.close()


if __name__ == "__main__":
    main()
