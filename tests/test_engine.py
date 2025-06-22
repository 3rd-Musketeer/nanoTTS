import pytest

from nanotts.audio_data import AudioSpec, UnsupportedFormat
from nanotts.engine import CallableEngine


def test_audio_spec_equality():
    """Test AudioSpec equality comparison."""
    spec1 = AudioSpec("pcm", 16000, 1, 16)
    spec2 = AudioSpec("pcm", 16000, 1, 16)
    spec3 = AudioSpec("mp3", 16000, 1, None)

    assert spec1 == spec2
    assert spec1 != spec3


@pytest.mark.asyncio
async def test_callable_engine():
    """Test CallableEngine wrapper."""

    def dummy_tts(text: str) -> bytes:
        return f"audio:{text}".encode()

    spec = AudioSpec("pcm", 16000, 1, 16)
    engine = CallableEngine(dummy_tts, output_spec=spec)

    chunk = await engine.synth("hello")

    assert chunk.data == b"audio:hello"
    assert chunk.spec == spec


def test_unsupported_format_exception():
    """Test UnsupportedFormat exception."""
    source = AudioSpec("mp3", 24000, 1, None)
    target = AudioSpec("pcm", 16000, 1, 16)

    exc = UnsupportedFormat(source, target)

    assert exc.source == source
    assert exc.target == target
    assert "mp3" in str(exc)
    assert "pcm" in str(exc)
