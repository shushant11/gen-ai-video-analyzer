from utils.files import cleanup_chunks


def test_cleanup_removes_chunks_and_their_source_wav(tmp_path):
    wav = tmp_path / "talk_converted.wav"
    wav.write_bytes(b"x")
    chunks = []
    for i in range(2):
        c = tmp_path / f"talk_converted.wav_chunk_{i}.wav"
        c.write_bytes(b"x")
        chunks.append(str(c))
    original = tmp_path / "talk.mp3"
    original.write_bytes(b"x")

    cleanup_chunks(chunks)

    assert not wav.exists()
    assert not any(tmp_path.glob("*_chunk_*"))
    assert original.exists()  # the user's own file is never touched


def test_cleanup_ignores_missing_files(tmp_path):
    cleanup_chunks([str(tmp_path / "gone.wav_chunk_0.wav")])  # must not raise
