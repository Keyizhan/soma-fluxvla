import json
import pickle

import numpy as np
import pytest

from fluxvla.datasets.cache import VideoCache


def make_cache(tmp_path, legacy=False):
    source = tmp_path / "data/videos/front/episode.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"test source identity")
    directory = tmp_path / "cache"
    directory.mkdir()
    frames = np.arange(5 * 4 * 6 * 3, dtype=np.uint8).reshape(5, 4, 6, 3)
    np.save(directory / "frames.npy", frames)
    pts = np.array([0.0, 0.04, 0.11, 0.17, 0.25])
    np.save(directory / "pts.npy", pts)
    stat = source.stat()
    if legacy:
        index = {str(source): {"cache": str(directory / "frames.npy"), "n": 5, "fps": 30.0}}
    else:
        index = {
            "version": 2,
            "entries": {
                "videos/front/episode.mp4": {
                    "cache": "frames.npy",
                    "timestamps": "pts.npy",
                    "n": 5,
                    "source": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                }
            },
        }
    (directory / "index.json").write_text(json.dumps(index))
    return source, directory, frames


def test_vfr_matching_order_duplicates_and_copy(tmp_path):
    source, directory, frames = make_cache(tmp_path)
    cache = VideoCache(directory, source.parents[2])
    result = cache.read(source, [0.16, 0.04, 0.16], tolerance_s=0.02)
    np.testing.assert_array_equal(result, frames[[3, 1, 3]].transpose(0, 3, 1, 2))
    assert result.flags.c_contiguous
    result[:] = 0
    np.testing.assert_array_equal(cache.read(source, [0.04])[0], frames[1].transpose(2, 0, 1))
    cache.close()


def test_out_of_range_and_stale_source_fail(tmp_path):
    source, directory, _ = make_cache(tmp_path)
    cache = VideoCache(directory, source.parents[2])
    with pytest.raises(ValueError, match="tolerance"):
        cache.read(source, [100])
    source.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="Stale"):
        cache.read(source, [0])
    cache.close()


def test_legacy_cloud_index_and_spawn_serialization(tmp_path):
    source, directory, frames = make_cache(tmp_path, legacy=True)
    cache = VideoCache(directory)
    np.testing.assert_array_equal(cache.read(source, [2 / 30])[0], frames[2].transpose(2, 0, 1))
    restored = pickle.loads(pickle.dumps(cache))
    assert not restored._arrays
    np.testing.assert_array_equal(restored.read(source, [0])[0], frames[0].transpose(2, 0, 1))
    cache.close()
    restored.close()


def test_mapping_limit_and_missing_entry(tmp_path):
    source, directory, _ = make_cache(tmp_path)
    index = json.loads((directory / "index.json").read_text())
    other = source.with_name("second.mp4")
    other.write_bytes(source.read_bytes())
    e = dict(index["entries"]["videos/front/episode.mp4"])
    e["source"] = {"size": other.stat().st_size, "mtime_ns": other.stat().st_mtime_ns}
    e["cache"] = "other.npy"
    (directory / "other.npy").write_bytes((directory / "frames.npy").read_bytes())
    index["entries"]["videos/front/second.mp4"] = e
    (directory / "index.json").write_text(json.dumps(index))
    cache = VideoCache(directory, source.parents[2], max_open_files=1)
    cache.read(source, [0])
    cache.read(other, [0])
    assert len(cache._arrays) == 1
    with pytest.raises(KeyError, match="absent"):
        cache.read(source.with_name("missing.mp4"), [0])
    cache.close()


def test_build_resume_and_pixel_identity(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("decord")
    from fluxvla.datasets.cache.build_video_cache import build_cache

    source = tmp_path / "data/videos/front/episode.mp4"
    source.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    assert writer.isOpened()
    for i in range(43):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 5
        frame[:, :, 1] = np.arange(64, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    directory = tmp_path / "cache"
    entries = build_cache(
        source.parents[2], directory, height=8, width=12, workers=1, chunk_frames=3
    )
    entry = next(iter(entries.values()))
    assert entry["codec"] == "zstd" and entry["block_frames"] == 8
    mtime = (directory / entry["cache"]).stat().st_mtime_ns
    assert build_cache(source.parents[2], directory, height=8, width=12, workers=1) == entries
    assert (directory / entry["cache"]).stat().st_mtime_ns == mtime
    import decord

    reader = decord.VideoReader(str(source), num_threads=1)
    cache = VideoCache(directory, source.parents[2])
    for i in (0, 7, 8, 17, 42):
        ts = float(reader.get_frame_timestamp(i)[0])
        expected = cv2.resize(reader[i].asnumpy(), (12, 8), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(cache.read(source, [ts])[0].transpose(1, 2, 0), expected)
    indices = [42, 7, 8, 7, 0]
    timestamps = [float(reader.get_frame_timestamp(i)[0]) for i in indices]
    expected = np.stack(
        [cv2.resize(reader[i].asnumpy(), (12, 8), interpolation=cv2.INTER_LINEAR) for i in indices]
    )
    np.testing.assert_array_equal(cache.read(source, timestamps).transpose(0, 2, 3, 1), expected)
    cache.close()
    # Re-running the builder repairs corrupt blocks, rather than trusting file existence.
    with (directory / entry["cache"]).open("r+b") as stream:
        stream.write(b"broken")
    repaired = build_cache(source.parents[2], directory, height=8, width=12, workers=1)
    assert repaired == entries
    cache = VideoCache(directory, source.parents[2])
    np.testing.assert_array_equal(cache.read(source, timestamps).transpose(0, 2, 3, 1), expected)
    cache.close()
    # A different resize setting must not silently reuse an incompatible cache.
    rebuilt = build_cache(source.parents[2], directory, height=10, width=10, workers=1)
    assert next(iter(rebuilt.values()))["cache"] != entry["cache"]
