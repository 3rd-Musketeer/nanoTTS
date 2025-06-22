"""
Tests for audio_data module - AudioSpec, AudioChunk, and AudioTranscoder.
"""

from unittest.mock import patch

import pytest

from nanotts.audio_data import AudioChunk, AudioSpec, AudioTranscoder, UnsupportedFormat


class TestAudioSpec:
    """Test AudioSpec data structure."""

    def test_creation(self):
        """Test AudioSpec creation."""
        spec = AudioSpec("pcm", 16000, 1, 16)
        assert spec.codec == "pcm"
        assert spec.sample_rate == 16000
        assert spec.channels == 1
        assert spec.sample_width == 16

    def test_equality(self):
        """Test AudioSpec equality comparison."""
        spec1 = AudioSpec("pcm", 16000, 1, 16)
        spec2 = AudioSpec("pcm", 16000, 1, 16)
        spec3 = AudioSpec("mp3", 16000, 1, None)

        assert spec1 == spec2
        assert spec1 != spec3

    def test_optional_sample_width(self):
        """Test AudioSpec with optional sample_width."""
        spec = AudioSpec("mp3", 24000, 2)
        assert spec.codec == "mp3"
        assert spec.sample_rate == 24000
        assert spec.channels == 2
        assert spec.sample_width is None


class TestAudioChunk:
    """Test AudioChunk data structure."""

    def test_creation(self):
        """Test AudioChunk creation."""
        spec = AudioSpec("pcm", 16000, 1, 16)
        chunk = AudioChunk(b"test_data", spec)

        assert chunk.data == b"test_data"
        assert chunk.spec == spec

    def test_with_different_specs(self):
        """Test AudioChunk with different specs."""
        pcm_spec = AudioSpec("pcm", 16000, 1, 16)
        mp3_spec = AudioSpec("mp3", 24000, 2, None)

        pcm_chunk = AudioChunk(b"pcm_data", pcm_spec)
        mp3_chunk = AudioChunk(b"mp3_data", mp3_spec)

        assert pcm_chunk.spec.codec == "pcm"
        assert mp3_chunk.spec.codec == "mp3"
        assert pcm_chunk.data != mp3_chunk.data


class TestUnsupportedFormat:
    """Test UnsupportedFormat exception."""

    def test_exception_creation(self):
        """Test UnsupportedFormat exception."""
        source = AudioSpec("mp3", 24000, 1, None)
        target = AudioSpec("pcm", 16000, 1, 16)

        exc = UnsupportedFormat(source, target)

        assert exc.source == source
        assert exc.target == target
        assert "mp3" in str(exc)
        assert "pcm" in str(exc)


class TestAudioTranscoder:
    """Test AudioTranscoder functionality."""

    def test_no_conversion_needed(self):
        """Test when no conversion is needed."""
        spec = AudioSpec("pcm", 16000, 1, 16)
        chunk = AudioChunk(b"test_data", spec)

        # Should return same chunk when specs match
        result = pytest.importorskip("asyncio").run(
            AudioTranscoder.convert(chunk, spec)
        )

        assert result is chunk

    def test_is_ffmpeg_available(self):
        """Test ffmpeg availability detection."""
        # Test when ffmpeg is available
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert AudioTranscoder.is_ffmpeg_available() is True

        # Test when ffmpeg is not available
        with patch("shutil.which", return_value=None):
            assert AudioTranscoder.is_ffmpeg_available() is False

    def test_get_ffmpeg_format_pcm(self):
        """Test ffmpeg format detection for PCM."""
        spec16 = AudioSpec("pcm", 16000, 1, 16)
        spec24 = AudioSpec("pcm", 16000, 1, 24)
        spec32 = AudioSpec("pcm", 16000, 1, 32)

        assert AudioTranscoder.get_ffmpeg_format(spec16) == "s16le"
        assert AudioTranscoder.get_ffmpeg_format(spec24) == "s24le"
        assert AudioTranscoder.get_ffmpeg_format(spec32) == "s32le"

    def test_get_ffmpeg_format_compressed(self):
        """Test ffmpeg format detection for compressed formats."""
        mp3_spec = AudioSpec("mp3", 24000, 1, None)
        opus_spec = AudioSpec("opus", 48000, 1, None)

        assert AudioTranscoder.get_ffmpeg_format(mp3_spec) == "mp3"
        assert AudioTranscoder.get_ffmpeg_format(opus_spec) == "opus"

    def test_get_ffmpeg_format_unsupported(self):
        """Test ffmpeg format detection for unsupported formats."""
        unknown_spec = AudioSpec("unknown", 16000, 1, 16)
        assert AudioTranscoder.get_ffmpeg_format(unknown_spec) is None

    @pytest.mark.asyncio
    async def test_conversion_no_ffmpeg_raises_error(self):
        """Test conversion fails when ffmpeg not available."""
        source_spec = AudioSpec("mp3", 24000, 1, None)
        target_spec = AudioSpec("pcm", 16000, 1, 16)
        chunk = AudioChunk(b"fake_mp3_data", source_spec)

        # Mock no ffmpeg available and no ffmpeg-python
        with (
            patch("shutil.which", return_value=None),
            patch(
                "nanotts.audio_data.AudioTranscoder.is_ffmpeg_available",
                return_value=False,
            ),
        ):
            with pytest.raises(UnsupportedFormat) as exc_info:
                await AudioTranscoder.convert(chunk, target_spec)

            assert exc_info.value.source == source_spec
            assert exc_info.value.target == target_spec

    @pytest.mark.asyncio
    async def test_ffmpeg_python_import_error_fallback(self):
        """Test fallback to manual ffmpeg when ffmpeg-python not available."""
        source_spec = AudioSpec("pcm", 16000, 1, 16)
        target_spec = AudioSpec("pcm", 22050, 1, 16)
        chunk = AudioChunk(b"test_pcm_data", source_spec)

        # Mock ffmpeg-python import error but manual ffmpeg fails too
        with (
            patch.dict("sys.modules", {"ffmpeg": None}),
            patch(
                "nanotts.audio_data.AudioTranscoder.is_ffmpeg_available",
                return_value=False,
            ),
        ):
            # Should fail with UnsupportedFormat when no conversion method available
            with pytest.raises(UnsupportedFormat):
                await AudioTranscoder.convert(chunk, target_spec)

    @pytest.mark.skipif(
        not AudioTranscoder.is_ffmpeg_available(), reason="ffmpeg not available"
    )
    @pytest.mark.asyncio
    async def test_manual_ffmpeg_conversion(self):
        """Test manual ffmpeg conversion (requires actual ffmpeg)."""
        # Create test PCM data (silence)
        pcm_data = b"\x00\x00" * 1600  # 100ms of 16-bit PCM at 16kHz
        source_spec = AudioSpec("pcm", 16000, 1, 16)
        target_spec = AudioSpec("pcm", 8000, 1, 16)  # Downsample
        chunk = AudioChunk(pcm_data, source_spec)

        # Force use of manual ffmpeg by mocking ffmpeg-python import failure
        with patch.dict("sys.modules", {"ffmpeg": None}):
            result = await AudioTranscoder.convert(chunk, target_spec)

            assert isinstance(result, AudioChunk)
            assert result.spec == target_spec
            assert len(result.data) > 0
            # Should be roughly half the length due to downsampling
            assert len(result.data) < len(pcm_data)
