"""Compare cached pixels against decoded, resized MP4 frames."""

import argparse
import random
from pathlib import Path

import numpy as np

from fluxvla.datasets.cache import VideoCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--videos", type=int, default=12)
    parser.add_argument("--frames-per-video", type=int, default=5)
    args = parser.parse_args()
    import cv2
    import decord

    cache = VideoCache(args.cache_dir, args.data_root)
    rng = random.Random(417)
    keys = sorted(cache.entries)
    if args.videos < 1 or args.frames_per_video < 1 or not keys:
        parser.error("Positive sample counts and a nonempty cache are required")
    checked = 0
    try:
        for key in rng.sample(keys, min(args.videos, len(keys))):
            source = Path(args.data_root) / key if cache.version == 2 else Path(key)
            reader = decord.VideoReader(str(source), num_threads=1)
            entry = cache.entries[key]
            if len(reader) != entry["n"]:
                raise AssertionError(f"Frame count mismatch: {source}")
            for index in rng.sample(range(len(reader)), min(args.frames_per_video, len(reader))):
                ts = float(np.asarray(reader.get_frame_timestamp(index))[0])
                actual = cache.read(source, [ts])[0].transpose(1, 2, 0)
                expected = reader[index].asnumpy()
                if expected.shape != actual.shape:
                    expected = cv2.resize(
                        expected, (actual.shape[1], actual.shape[0]), interpolation=cv2.INTER_LINEAR
                    )
                np.testing.assert_array_equal(actual, expected, err_msg=str(source))
                checked += 1
    finally:
        cache.close()
    print(f"PASS: {checked} decoded/resized frames are pixel-identical")


if __name__ == "__main__":
    main()
