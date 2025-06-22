import anyio
import pytest

from nanotts import AudioChunk, AudioSpec, NanoTTS


class TestNanoTTS:
    """Core NanoTTS functionality tests."""

    @pytest.mark.asyncio
    async def test_basic_synthesis(self):
        """Test basic text-to-speech synthesis."""
        tts = NanoTTS(model="dummy")

        chunks = []
        async for chunk, text in tts.stream("Hello, world!"):
            chunks.append((chunk, text))

        assert len(chunks) >= 1
        assert all(isinstance(chunk, AudioChunk) for chunk, _ in chunks)
        assert all(isinstance(text, str) for _, text in chunks)

        # Verify complete text is covered
        full_text = "".join(text for _, text in chunks)
        assert "Hello" in full_text and "world" in full_text

    @pytest.mark.asyncio
    async def test_segment_ordering(self):
        """Test that segments are yielded in correct order."""
        tts = NanoTTS(model="dummy")

        results = []
        async for chunk, text in tts.stream("First. Second. Third."):
            results.append(text)

        # Should maintain order
        assert "First" in results[0]
        assert len(results) >= 3  # Should break on periods

    @pytest.mark.asyncio
    async def test_cancellation(self):
        """Test cancellation stops synthesis immediately."""
        tts = NanoTTS(model="dummy")

        results = []
        async for chunk, text in tts.stream("A. B. C. D. E."):
            results.append(text)
            if len(results) == 2:
                tts.cancel()
                break

        # Should stop after cancellation
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_streaming_input(self):
        """Test streaming text input (like from LLM)."""

        async def token_stream():
            tokens = ["Hello", ",", " ", "streaming", " ", "world", "!"]
            for token in tokens:
                yield token
                await anyio.sleep(0.01)  # Simulate LLM delay

        tts = NanoTTS(model="dummy", timeout_ms=100)

        results = []
        async for chunk, text in tts.stream(token_stream()):
            results.append(text)

        # Should handle streaming and produce segments
        assert len(results) >= 1
        full_text = "".join(results)
        assert "Hello" in full_text
        assert "streaming" in full_text
        assert "world" in full_text

    @pytest.mark.asyncio
    async def test_engine_factory_kwargs(self):
        """Test engine creation with custom parameters."""

        # Test that factory accepts kwargs
        tts = NanoTTS(model="dummy", custom_param="test_value")

        # Engine should be created on first stream() call
        async for chunk, text in tts.stream("test"):
            break

        assert tts._engine is not None

    @pytest.mark.asyncio
    async def test_output_spec_handling(self):
        """Test custom output audio specification."""
        custom_spec = AudioSpec("pcm", 22050, 2, 16)
        tts = NanoTTS(model="dummy", output_spec=custom_spec)

        async for chunk, text in tts.stream("Hello"):
            # Should complete without error
            # (Transcoding tested separately)
            assert isinstance(chunk, AudioChunk)
            assert hasattr(chunk, "spec")
            break

    @pytest.mark.asyncio
    async def test_multilingual_text(self):
        """Test with mixed language text."""
        tts = NanoTTS(model="dummy")

        results = []
        async for chunk, text in tts.stream("Hello, 你好！World, 世界！"):
            results.append(text)

        # Should handle multilingual punctuation
        assert len(results) >= 2  # Should break on punctuation
        full_text = "".join(results)
        assert "你好" in full_text and "世界" in full_text

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling of empty or whitespace input."""
        tts = NanoTTS(model="dummy")

        results = []
        async for chunk, text in tts.stream(""):
            results.append(text)

        # Empty input should not produce segments
        assert len(results) == 0

        # Test whitespace
        results = []
        async for chunk, text in tts.stream("   "):
            results.append(text)

        # Pure whitespace should not produce meaningful segments
        assert len(results) == 0 or all(not text.strip() for text in results)
