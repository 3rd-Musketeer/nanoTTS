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
    async def test_tiktoken_segmentation_core_logic(self):
        """Test core token-based segmentation logic."""
        tts = NanoTTS(model="dummy", min_tokens=3, max_tokens=15)

        # Test 1: Sentence boundary detection with sufficient tokens
        async def incremental_sentences():
            parts = ["Hello world. ", "How are you? ", "Fine thanks."]
            for part in parts:
                yield part
                await anyio.sleep(0.01)

        results = []
        async for chunk, text in tts.stream(incremental_sentences()):
            results.append(text)

        # Should break at sentence boundaries when streaming incrementally
        assert len(results) >= 2
        # Each segment should contain complete sentences
        for result in results:
            assert result.strip()  # No empty segments

        # Test 2: Token limit enforcement
        long_text = "This is a very long sentence that definitely exceeds our token limits and should be broken appropriately at word boundaries while respecting token counts."

        results2 = []
        async for chunk, text in tts.stream(long_text):
            results2.append(text)

        # Should break when exceeding max_tokens
        if len(results2) > 1:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            for segment in results2[:-1]:  # All but last
                tokens = len(enc.encode(segment))
                assert tokens <= 15  # Should respect max_tokens

    @pytest.mark.asyncio
    async def test_abbreviation_preservation(self):
        """Test that abbreviations like Ph.D don't get broken incorrectly."""
        tts = NanoTTS(model="dummy", min_tokens=3, max_tokens=20)

        # Test the original problem case
        test_cases = [
            "Dr. Smith has a Ph.D. in Computer Science.",
            "The U.S.A. is a large country with many Ph.D. programs.",
            "Students study for 3.5 years on average, earning $45.99 per hour.",
            "Version 2.1 includes updates to section 1.5 and appendix A.3.",
        ]

        for text in test_cases:
            results = []
            async for chunk, segment in tts.stream(text):
                results.append(segment)

            full_text = " ".join(results)

            # Key test: abbreviations should stay together
            if "Ph.D." in text:
                assert "Ph.D." in full_text, f"Ph.D. was broken in: {full_text}"
            if "U.S.A." in text:
                assert "U.S.A." in full_text, f"U.S.A. was broken in: {full_text}"
            if "3.5" in text:
                assert "3.5" in full_text, f"3.5 was broken in: {full_text}"
            if "$45.99" in text:
                assert "$45.99" in full_text, f"$45.99 was broken in: {full_text}"

    @pytest.mark.asyncio
    async def test_markdown_preprocessing(self):
        """Test that markdown is properly cleaned before segmentation."""
        tts = NanoTTS(model="dummy", min_tokens=3, max_tokens=20)

        markdown_cases = [
            ("### What is a **Ph.D**?", "What is a Ph.D?"),
            ("The `code` and *italic* text.", "The code and italic text."),
            ("[Click here](http://example.com) for more.", "Click here for more."),
            ("> This is a quote with **bold**.", "This is a quote with bold."),
            ("- Item 1\n- Item 2\n- Item 3", "Item 1 Item 2 Item 3"),
        ]

        for markdown_input, expected_clean in markdown_cases:
            results = []
            async for chunk, segment in tts.stream(markdown_input):
                results.append(segment)

            full_text = " ".join(results)

            # Check markdown was cleaned
            assert "**" not in full_text, f"Bold markdown not cleaned: {full_text}"
            assert "###" not in full_text, f"Header markdown not cleaned: {full_text}"
            assert "`" not in full_text, f"Code markdown not cleaned: {full_text}"
            assert "[" not in full_text or "]" not in full_text, (
                f"Link markdown not cleaned: {full_text}"
            )

            # Check key content preserved
            if "Ph.D" in expected_clean:
                assert "Ph.D" in full_text, f"Ph.D not preserved: {full_text}"

    @pytest.mark.asyncio
    async def test_cancellation_stops_processing(self):
        """Test cancellation stops further processing."""
        tts = NanoTTS(model="dummy", min_tokens=1, max_tokens=10)

        # Test cancellation after getting some output
        async def slow_text():
            parts = ["Hello world. ", "This should be cancelled."]
            for i, part in enumerate(parts):
                yield part
                await anyio.sleep(0.01)
                if i == 0:  # Cancel after first part is yielded
                    tts.cancel()

        results = []
        async for chunk, text in tts.stream(slow_text()):
            results.append(text)

        # Should have at least some output before cancellation
        assert len(results) >= 1
        # Cancelled text should not appear in results
        full_text = " ".join(results)
        assert "cancelled" not in full_text.lower()

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
        """Test streaming API handles different input types."""
        tts = NanoTTS(model="dummy", min_tokens=2, max_tokens=15, timeout_ms=800)

        # Test 1: String input
        chunks = []
        async for chunk, text in tts.stream("Hello world"):
            chunks.append((chunk, text))
        assert len(chunks) >= 1

        # Test 2: Async iterable input (preserves spaces properly)
        async def text_tokens():
            for token in ["Hello", " ", "world"]:
                yield token
                await anyio.sleep(0.01)

        chunks = []
        async for chunk, text in tts.stream(text_tokens()):
            chunks.append((chunk, text))

        # Should handle streaming input and preserve spaces
        assert len(chunks) >= 1
        full_text = "".join(text for _, text in chunks)
        assert "Hello" in full_text and "world" in full_text

    def test_model_manager_singleton(self):
        """Test ModelManager singleton pattern from dev docs."""
        from nanotts.model import manager as manager1
        from nanotts.model import manager as manager2

        # Should be same instance
        assert manager1 is manager2

        # Should have dummy registered
        models = manager1.list_models()
        assert "dummy" in models


class TestTiktokenRegressionPrevention:
    """Prevent regressions - focus on the original problems we solved."""

    @pytest.mark.asyncio
    async def test_phd_abbreviation_not_broken(self):
        """Prevent regression: Ph.D should never be broken into Ph. + D."""
        tts = NanoTTS(model="dummy", min_tokens=3, max_tokens=20)

        # The original failing case from the user's example
        problematic_text = "what is a Ph.D? A Ph.D. (Doctor of Philosophy) is the highest academic degree"

        results = []
        async for chunk, text in tts.stream(problematic_text):
            results.append(text)

        full_text = " ".join(results)

        # Critical test: Ph.D should never be broken
        assert "Ph.D" in full_text, f"Ph.D was broken or missing: {full_text}"
        assert "Ph." not in full_text or "Ph.D." in full_text, (
            f"Ph.D was improperly broken: {full_text}"
        )

    @pytest.mark.asyncio
    async def test_numbers_and_decimals_preserved(self):
        """Prevent regression: numbers like 3.14, $5.99 should stay intact."""
        tts = NanoTTS(model="dummy", min_tokens=3, max_tokens=20)

        test_cases = [
            "The value of pi is 3.14159 approximately.",
            "It costs $45.99 per month for the service.",
            "Version 2.1 includes updates to section 1.5.",
            "The temperature was 98.6 degrees Fahrenheit.",
        ]

        for text in test_cases:
            results = []
            async for chunk, segment in tts.stream(text):
                results.append(segment)

            full_text = " ".join(results)

            # Extract numbers from original text
            import re

            numbers = re.findall(r"\$?\d+\.\d+", text)

            for number in numbers:
                assert number in full_text, (
                    f"Number {number} was broken in: {full_text}"
                )

    @pytest.mark.asyncio
    async def test_markdown_cleaning_regression(self):
        """Prevent regression: markdown should be cleaned consistently."""
        tts = NanoTTS(model="dummy", min_tokens=3, max_tokens=20)

        # Test various markdown formats that could cause issues
        markdown_inputs = [
            "### Key Characteristics of a Ph.D.:",
            "**Advanced Research**: A Ph.D. program typically involves...",
            "The `process()` function returns a Ph.D. candidate.",
            "> Quote: Dr. Smith has a Ph.D. in Computer Science.",
        ]

        for markdown_text in markdown_inputs:
            results = []
            async for chunk, segment in tts.stream(markdown_text):
                results.append(segment)

            full_text = " ".join(results)

            # Ensure markdown symbols are removed
            assert "**" not in full_text, f"Bold markdown not cleaned: {full_text}"
            assert "###" not in full_text, f"Header markdown not cleaned: {full_text}"
            assert "`" not in full_text, f"Code markdown not cleaned: {full_text}"

            # Ensure Ph.D. is preserved if present
            if "Ph.D" in markdown_text:
                assert "Ph.D" in full_text, (
                    f"Ph.D not preserved after markdown cleaning: {full_text}"
                )
