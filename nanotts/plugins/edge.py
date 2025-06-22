"""
Edge-TTS plugin for nanoTTS

Uses Microsoft Edge's online text-to-speech service.
No API key required.
"""

from __future__ import annotations

import edge_tts

from ..audio_data import AudioChunk, AudioSpec
from ..engine import Engine
from ..model import manager


class EdgeEngine(Engine):
    def __init__(
        self,
        voice: str = "en-AU-NatashaNeural",  # Use a working voice
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

        # Edge-TTS outputs MP3 format
        self._output_spec = AudioSpec("mp3", 24000, 1, None)

    async def synth(self, text: str, *, target: AudioSpec | None = None) -> AudioChunk:
        """Synthesize text using Edge-TTS."""
        if not text.strip():
            return AudioChunk(b"", self._output_spec)

        try:
            # Create communicate object with voice settings
            communicate = edge_tts.Communicate(
                text, self.voice, rate=self.rate, volume=self.volume, pitch=self.pitch
            )

            # Generate audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            return AudioChunk(audio_data, self._output_spec)

        except Exception as e:
            # Handle network errors, service unavailable, etc.
            raise RuntimeError(f"Edge-TTS synthesis failed: {e}") from e


async def build_edge(
    *,
    voice: str = "en-AU-NatashaNeural",  # Use a working voice
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    **kwargs,
) -> EdgeEngine:
    """Build Edge-TTS engine with specified parameters."""
    return EdgeEngine(voice=voice, rate=rate, volume=volume, pitch=pitch)


# Register the edge engine
manager.register(
    "edge",
    build_edge,
    "edge: voice=<voice_name>, rate=<±N%>, volume=<±N%>, pitch=<±NHz> (Microsoft Edge TTS)",
)
