"""
Plugin system tests for nanoTTS.
Tests engine registration and optional plugins.
"""

import pytest


class TestPluginSystem:
    """Test plugin registration and loading."""

    def test_dummy_plugin_available(self):
        """Test that dummy plugin is always available."""
        from nanotts.model import manager

        models = manager.list_models()
        assert "dummy" in models
        assert "dummy" in models["dummy"].lower()

    @pytest.mark.asyncio
    async def test_dummy_engine_creation(self):
        """Test dummy engine can be created and used."""
        from nanotts.model import manager

        engine = await manager.get("dummy")
        assert engine is not None

        chunk = await engine.synth("test")
        assert chunk.data != b""
        assert chunk.spec.codec == "pcm"

    def test_edge_plugin_conditional(self):
        """Test edge plugin availability based on dependencies."""
        from nanotts.model import manager

        models = manager.list_models()

        try:
            import edge_tts

            # If edge-tts is available, plugin should be registered
            assert "edge" in models
            assert (
                "edge" in models["edge"].lower()
                or "microsoft" in models["edge"].lower()
            )
        except ImportError:
            # If edge-tts not available, should not be registered
            assert "edge" not in models

    @pytest.mark.asyncio
    async def test_engine_factory_pattern(self):
        """Test engine factory pattern with kwargs."""
        from nanotts.model import manager

        # Test factory accepts arbitrary kwargs
        engine1 = await manager.get("dummy", param1="value1")
        engine2 = await manager.get("dummy", param2="value2", param3=123)

        assert engine1 is not None
        assert engine2 is not None
        # Different kwargs should create different instances
        assert engine1 is not engine2

    def test_plugin_import_safety(self):
        """Test that plugin imports don't crash on missing deps."""
        # These should not raise ImportError
        try:
            import nanotts.plugins
            import nanotts.plugins.dummy
        except ImportError:
            pytest.fail("Core plugins should always be importable")

        # Edge plugin should handle missing dependencies gracefully
        try:
            import nanotts.plugins.edge
        except ImportError:
            # This is expected if edge-tts not installed
            pass


class TestEdgePlugin:
    """Test Edge-TTS plugin when available."""

    @pytest.fixture
    def edge_tts(self):
        """Skip tests if edge-tts not available."""
        return pytest.importorskip("edge_tts")

    def test_edge_registration(self, edge_tts):
        """Test edge plugin registration."""
        from nanotts.model import manager

        models = manager.list_models()
        assert "edge" in models

    @pytest.mark.asyncio
    async def test_edge_factory(self, edge_tts):
        """Test edge engine factory without network calls."""
        from nanotts.model import manager

        # Test factory can create engine with custom params
        engine = await manager.get("edge", voice="en-US-AriaNeural", rate="+10%")

        assert hasattr(engine, "voice")
        assert engine.voice == "en-US-AriaNeural"
        assert hasattr(engine, "rate")
        assert engine.rate == "+10%"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network connection to Microsoft Edge-TTS")
    async def test_edge_synthesis(self, edge_tts):
        """Test actual edge synthesis (network required)."""
        from nanotts.model import manager

        engine = await manager.get("edge")
        chunk = await engine.synth("Hello world")

        assert chunk.data != b""
        assert chunk.spec.codec == "mp3"
        assert chunk.spec.sample_rate == 24000

    @pytest.mark.asyncio
    async def test_edge_empty_text(self, edge_tts):
        """Test edge engine with empty text."""
        from nanotts.model import manager

        engine = await manager.get("edge")
        chunk = await engine.synth("")

        assert chunk.data == b""
        assert chunk.spec.codec == "mp3"
