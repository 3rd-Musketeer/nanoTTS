from ..audio_data import AudioSpec
from ..engine import CallableEngine
from ..model import manager


async def build_dummy(**kwargs) -> CallableEngine:
    """Build a dummy TTS engine for testing purposes."""

    def _dummy_synth(text: str) -> bytes:
        # Generate predictable dummy audio data based on text length
        audio_data = f"DUMMY_AUDIO[{text}]".encode() * (len(text) // 10 + 1)
        return audio_data[: len(text) * 16]  # 16 bytes per character

    spec = AudioSpec("pcm", 16000, 1, 16)
    return CallableEngine(_dummy_synth, output_spec=spec)


# Register the dummy engine
manager.register(
    "dummy",
    build_dummy,
    "dummy: simple test engine that returns predictable audio data",
)
