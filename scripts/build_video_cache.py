"""Build 8-frame Zstd training caches without loading a whole video into RAM."""

import argparse

from fluxvla.datasets.cache.build_video_cache import build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--compression-level", type=int, default=3)
    args = parser.parse_args()
    entries = build_cache(
        args.data_root,
        args.cache_dir,
        args.height,
        args.width,
        args.workers,
        args.chunk_frames,
        args.compression_level,
    )
    print(f"Ready: {len(entries)} videos, {sum(e['n'] for e in entries.values())} frames")


if __name__ == "__main__":
    main()
