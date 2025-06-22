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
        """Test that segments are yielded in correct order with token-based logic."""
        tts = NanoTTS(model="dummy", min_tokens=2, max_tokens=15)

        # Test with incremental input for better segmentation
        async def incremental_text():
            parts = ["First. ", "Second. ", "Third."]
            for part in parts:
                yield part
                await anyio.sleep(0.01)

        results = []
        async for chunk, text in tts.stream(incremental_text()):
            results.append(text)

        # Should maintain order and handle sentence boundaries
        assert len(results) >= 1
        full_text = " ".join(results)
        assert "First" in full_text
        # Check order is preserved
        first_pos = full_text.find("First")
        second_pos = full_text.find("Second")
        third_pos = full_text.find("Third")
        if second_pos >= 0:
            assert first_pos < second_pos
        if third_pos >= 0 and second_pos >= 0:
            assert second_pos < third_pos

    @pytest.mark.asyncio
    async def test_cancellation(self):
        """Test cancellation stops synthesis immediately."""
        tts = NanoTTS(model="dummy", min_tokens=1, max_tokens=10)

        # Use incremental input to better test cancellation
        async def slow_text():
            parts = ["Hello", " world", ". More", " text", " here."]
            for i, part in enumerate(parts):
                if i == 2:  # Cancel after "Hello world. More"
                    tts.cancel()
                yield part
                await anyio.sleep(0.01)

        results = []
        async for chunk, text in tts.stream(slow_text()):
            results.append(text)

        # Should stop after cancellation
        full_text = " ".join(results)
        assert "text here" not in full_text  # Should not process cancelled text

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
        """Test with mixed language text and consistent token counting."""
        tts = NanoTTS(model="dummy", min_tokens=3, max_tokens=20)

        # Test that token counting works consistently across languages
        results = []
        async for chunk, text in tts.stream("Hello world. 你好世界。How are you?"):
            results.append(text)

        # Should handle multilingual text appropriately
        assert len(results) >= 1
        full_text = " ".join(results)
        assert "你好" in full_text and "世界" in full_text

        # Test token counting consistency
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        # English and Chinese should be counted consistently
        en_text = "Hello world"
        zh_text = "你好世界"

        en_tokens = len(enc.encode(en_text))
        zh_tokens = len(enc.encode(zh_text))

        # Both should be tokenized (this is mainly to verify tiktoken works)
        assert en_tokens > 0 and zh_tokens > 0

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
