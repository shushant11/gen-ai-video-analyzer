import os


def cleanup_chunks(chunk_paths):
    """Delete the chunk files made by chunk_audio() and the WAV they were cut from.
    Chunks are named "<wav>_chunk_<n>.wav", so the source WAV is everything before "_chunk_"."""
    wavs = set()
    for path in chunk_paths:
        wavs.add(path.rsplit("_chunk_", 1)[0])
        _remove(path)
    for wav in wavs:
        _remove(wav)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
