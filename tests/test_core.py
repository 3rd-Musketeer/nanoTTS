"""
Core integration tests for nanoTTS.
Tests the essential functionality described in the dev docs.
"""

import anyio
import pytest

from nanotts import AudioChunk, AudioSpec, NanoTTS
from nanotts.audio_data import UnsupportedFormat
from nanotts.engine import CallableEngine
from nanotts.model import manager


class TestCoreIntegration:
    """Essential integration tests from dev docs requirements."""

    @pytest.mark.asyncio
    async def test_dev_spec_segment_cutting(self):
        """Test segment cutting as specified in dev docs."""
        tts = NanoTTS(model="dummy", max_len=30)

        # Test 1: Strong punctuation split
        results = []
        async for chunk, text in tts.stream("A。B！C？"):
            results.append(text)

        assert len(results) == 3
        assert results == ["A。", "B！", "C？"]

        # Test 2: Max length split with smart breaking
        long_text = (
            "This is a very long sentence that should be split at word boundaries"
        )
        results = []
        async for chunk, text in tts.stream(long_text):
            results.append(text)

        assert len(results) > 1
        # Should not break mid-word
        for result in results[:-1]:
            words = result.strip().split()
            if words:
                # Last word should be complete (not end with partial word)
                assert len(words[-1]) > 1 or words[-1] in ".,!?;:"

    @pytest.mark.asyncio
    async def test_dev_spec_end_to_end_order(self):
        """Test end-to-end order requirement from dev docs."""
        tts = NanoTTS(model="dummy")

        results = []
        async for chunk, text in tts.stream("A。B。"):
            results.append(text)

        # Must yield A, then B in order
        assert results == ["A。", "B。"]

    @pytest.mark.asyncio
    async def test_dev_spec_cancellation(self):
        """Test cancellation requirement from dev docs."""
        tts = NanoTTS(model="dummy")

        results = []
        async for chunk, text in tts.stream("A。B。C。"):
            results.append(text)
            if len(results) == 1:
                tts.cancel()
                break

        # After cancel(), no further chunks should be emitted
        assert len(results) == 1
        assert results[0] == "A。"

    @pytest.mark.asyncio
    async def test_factory_kwargs_requirement(self):
        """Test factory kwargs as specified in dev docs v1.1."""

        # Test engine factory with kwargs
        engine = await manager.get("dummy", custom_param="test_value")
        assert engine is not None

        # Test NanoTTS with factory kwargs
        tts = NanoTTS(model="dummy", voice="test_voice", rate="+10%")

        async for chunk, text in tts.stream("test"):
            assert isinstance(chunk, AudioChunk)
            break

        assert tts._engine is not None

    def test_audio_spec_dataclass(self):
        """Test AudioSpec/AudioChunk data structures from dev docs."""
        # Test AudioSpec
        spec = AudioSpec("pcm", 16000, 1, 16)
        assert spec.codec == "pcm"
        assert spec.sample_rate == 16000
        assert spec.channels == 1
        assert spec.sample_width == 16

        # Test equality
        spec2 = AudioSpec("pcm", 16000, 1, 16)
        spec3 = AudioSpec("mp3", 16000, 1, None)
        assert spec == spec2
        assert spec != spec3

        # Test AudioChunk
        chunk = AudioChunk(b"test_data", spec)
        assert chunk.data == b"test_data"
        assert chunk.spec == spec

    @pytest.mark.asyncio
    async def test_callable_engine_wrapper(self):
        """Test CallableEngine wrapper from dev docs."""

        def dummy_synth(text: str) -> bytes:
            return f"audio_for_{text}".encode()

        spec = AudioSpec("pcm", 16000, 1, 16)
        engine = CallableEngine(dummy_synth, output_spec=spec)

        chunk = await engine.synth("hello")

        assert chunk.data == b"audio_for_hello"
        assert chunk.spec == spec

    @pytest.mark.asyncio
    async def test_minimal_transcoder_concept(self):
        """Test transcoding concept (UnsupportedFormat for missing ffmpeg)."""
        source_spec = AudioSpec("mp3", 24000, 1, None)
        target_spec = AudioSpec("pcm", 16000, 1, 16)

        # Test exception
        exc = UnsupportedFormat(source_spec, target_spec)
        assert exc.source == source_spec
        assert exc.target == target_spec
        assert "mp3" in str(exc) and "pcm" in str(exc)

    @pytest.mark.asyncio
    async def test_streaming_api_requirements(self):
        """Test streaming API as specified in dev docs."""
        tts = NanoTTS(model="dummy", timeout_ms=800)

        # Test 1: String input
        chunks = []
        async for chunk, text in tts.stream("Hello world"):
            chunks.append((chunk, text))
        assert len(chunks) >= 1

        # Test 2: Async iterable input
        async def text_tokens():
            for token in ["Hello", " ", "world"]:
                yield token
                await anyio.sleep(0.01)

        chunks = []
        async for chunk, text in tts.stream(text_tokens()):
            chunks.append((chunk, text))

        # Should handle streaming input
        assert len(chunks) >= 1
        full_text = "".join(text for _, text in chunks)
        assert full_text.strip() == "Hello world"

    def test_model_manager_singleton(self):
        """Test ModelManager singleton pattern from dev docs."""
        from nanotts.model import manager as manager1
        from nanotts.model import manager as manager2

        # Should be same instance
        assert manager1 is manager2

        # Should have dummy registered
        models = manager1.list_models()
        assert "dummy" in models


class TestRegressionPrevention:
    """Prevent regressions in recent fixes."""

    @pytest.mark.asyncio
    async def test_no_premature_timeout_breaks(self):
        """Prevent regression: timeout was triggering after every token."""
        tts = NanoTTS(model="dummy", timeout_ms=800)

        # Simulate token feeding like in demo
        async def llm_tokens():
            for token in ["I", " am", " Qwen", ",", " a", " large", " model"]:
                yield token
                await anyio.sleep(0.08)  # 80ms like real LLM

        results = []
        async for chunk, text in tts.stream(llm_tokens()):
            results.append(text)

        # Should NOT break into 7+ segments like before
        assert len(results) <= 3, f"Too many segments: {results}"
        # Should break properly at comma
        assert any("," in result for result in results)

    @pytest.mark.asyncio
    async def test_no_mid_word_breaks(self):
        """Prevent regression: words like 'answering' split as 'answerin' + 'g'."""
        tts = NanoTTS(model="dummy", max_len=50)

        text = "I am here to help with answering questions and providing information"
        results = []
        async for chunk, text in tts.stream(text):
            results.append(text)

        # Check no mid-word breaks
        problematic_patterns = ["answerin", "providin", "informatio"]
        for result in results:
            for pattern in problematic_patterns:
                if pattern in result and result.endswith(pattern[:-1]):
                    pytest.fail(
                        f"Found mid-word break: '{result}' ends with '{pattern[:-1]}'"
                    )

    @pytest.mark.asyncio
    async def test_punctuation_priority_over_length(self):
        """Prevent regression: punctuation should override max_len."""
        tts = NanoTTS(model="dummy", max_len=200)  # Very long max_len

        results = []
        async for chunk, text in tts.stream("Short sentence. Another short sentence."):
            results.append(text)

        # Should break at periods, not wait for max_len
        assert len(results) == 2
        assert results[0].endswith(".")
        assert results[1].endswith(".")
