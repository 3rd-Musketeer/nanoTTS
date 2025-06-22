#!/usr/bin/env python3
"""
Self-contained Interactive TTS Demo with LLM Integration

Commands:
- /mode: Switch between token-feeding and full-text synthesis modes
- /verbose: Toggle verbose output showing segmentation results
- /audio: Toggle audio playback on/off
- /language: Switch between English and Chinese voices
- /performance: Benchmark last response across all modes
- /help: Show commands cheatsheet
- /exit: Quit demo
"""

import asyncio
from collections.abc import AsyncIterator
from enum import Enum
import os
from pathlib import Path

# Import nanoTTS
import sys
import tempfile
import time
from typing import Any, Optional

import anyio
from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from nanotts import AudioChunk, AudioSpec, NanoTTS


class SynthMode(Enum):
    TOKEN_FEEDING = "token-feeding"
    FULL_TEXT = "full-text"


async def play_audio_chunk(chunk: AudioChunk, verbose: bool = False) -> None:
    """Play audio chunk using sounddevice."""
    try:
        # Convert audio chunk to numpy array for playback
        def convert_and_play(data: bytes, spec: AudioSpec) -> None:
            try:
                if spec.codec == "pcm":
                    # Direct PCM data
                    if spec.sample_width == 16:
                        dtype = np.int16
                        scale = 32768.0
                    elif spec.sample_width == 24 or spec.sample_width == 32:
                        dtype = np.int32
                        scale = 2147483648.0
                    else:
                        if verbose:
                            print(
                                f"   ⚠️ Unsupported PCM bit depth: {spec.sample_width}"
                            )
                        return

                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(data, dtype=dtype)

                    # Reshape for channels
                    if spec.channels > 1:
                        audio_array = audio_array.reshape(-1, spec.channels)

                    # Convert to float for sounddevice
                    audio_array = audio_array.astype(np.float32) / scale

                else:
                    # Use soundfile to decode compressed formats (MP3, etc.)
                    with tempfile.NamedTemporaryFile(
                        suffix=f".{spec.codec}"
                    ) as temp_file:
                        temp_file.write(data)
                        temp_file.flush()

                        audio_array, sample_rate = sf.read(temp_file.name)
                        if spec.sample_rate != sample_rate and verbose:
                            print(
                                f"   ⚠️ Sample rate mismatch: expected {spec.sample_rate}, got {sample_rate}"
                            )

                        audio_array = audio_array.astype(np.float32)

                # Play the audio
                sd.play(audio_array, samplerate=spec.sample_rate)
                sd.wait()  # Wait for playback to complete

            except Exception as e:
                if verbose:
                    print(f"   ❌ Audio conversion/playback error: {e}")

        # Run in thread pool to avoid blocking
        await anyio.to_thread.run_sync(convert_and_play, chunk.data, chunk.spec)

    except Exception as e:
        if verbose:
            print(f"   ⚠️ Audio playback error: {e}")


class InteractiveDemo:
    def __init__(self):
        self.console = Console()
        self.client: Optional[AsyncOpenAI] = None
        self.tts: Optional[NanoTTS] = None

        # Demo state
        self.mode = SynthMode.TOKEN_FEEDING
        self.verbose = True  # Default to verbose mode
        self.audio_enabled = True
        self.last_response = ""
        self.language = "en"  # Current language: "en" or "zh"

        # Voice configurations
        self.voices = {
            "en": "en-AU-NatashaNeural",  # Australian English
            "zh": "zh-CN-XiaoxiaoNeural",  # Mainland Chinese
        }

    def load_config(self) -> bool:
        """Load API configuration from .env file."""
        load_dotenv()

        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("MODEL", "gpt-4o-mini")

        if not api_key:
            self.console.print("[red]Error: API_KEY not found in .env file[/red]")
            self.console.print("Please copy .env.example to .env and add your API key")
            return False

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # Initialize TTS
        try:
            voice = self.voices[self.language]
            self.tts = NanoTTS(model="edge", voice=voice)
            lang_name = (
                "English (Australian)"
                if self.language == "en"
                else "Chinese (Mainland)"
            )
            self.console.print(
                f"[green]✓ Using Edge-TTS engine ({lang_name} voice)[/green]"
            )
        except (ValueError, ImportError):
            self.tts = NanoTTS(model="dummy")
            self.console.print(
                "[yellow]⚠ Using dummy engine (install edge-tts for real synthesis)[/yellow]"
            )

        status = "🔊 enabled" if self.audio_enabled else "🔇 disabled"
        self.console.print(f"[blue]Audio playback {status}[/blue]")
        return True

    def show_help(self):
        """Display command cheatsheet."""
        commands = [
            ("/mode", "Switch synthesis mode (token-feeding ↔ full-text)"),
            ("/verbose", "Toggle verbose segmentation output"),
            ("/audio", "Toggle audio playback on/off"),
            ("/language", "Switch between English (🇦🇺) and Chinese (🇨🇳) voices"),
            ("/performance", "Benchmark last response across all modes"),
            ("/help", "Show this help message"),
            ("/exit", "Quit demo"),
        ]

        table = Table(title="Interactive TTS Demo Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)
        self.console.print(f"\n[yellow]Mode:[/yellow] {self.mode.value}")
        self.console.print(
            f"[yellow]Verbose:[/yellow] {'ON' if self.verbose else 'OFF'}"
        )
        self.console.print(
            f"[yellow]Audio:[/yellow] {'ON' if self.audio_enabled else 'OFF'}"
        )
        lang_display = "English" if self.language == "en" else "Chinese"
        self.console.print(
            f"[yellow]Language:[/yellow] {lang_display} ({self.voices[self.language]})"
        )

    async def get_llm_response(self, user_input: str) -> AsyncIterator[str]:
        """Get streaming response from LLM."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_input}],
                stream=True,
                max_tokens=500,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.console.print(f"[red]LLM Error: {e}[/red]")
            yield "Sorry, I encountered an error processing your request."

    async def play_and_log(self, chunk: AudioChunk, text: str, segment_num: int):
        """Play audio and log if verbose."""
        if self.verbose:
            self.console.print(
                f'[dim]Segment {segment_num}: "{text}" ({len(text)} chars, {len(chunk.data)} bytes)[/dim]'
            )

        if self.audio_enabled:
            await play_audio_chunk(chunk, self.verbose)
        else:
            await anyio.sleep(0.1)  # Simulate playback timing

    async def synthesize_streaming(
        self, text_stream: AsyncIterator[str]
    ) -> dict[str, Any]:
        """Synthesize with token-feeding mode."""
        self.console.print("[blue]🎤 Synthesizing (token-feeding)...[/blue]")

        start_time = time.perf_counter()
        first_audio_time = None
        audio_chunks = 0

        async for chunk, text in self.tts.stream(text_stream):
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - start_time

            audio_chunks += 1
            await self.play_and_log(chunk, text, audio_chunks)

        total_time = time.perf_counter() - start_time
        return {
            "first_audio_latency": first_audio_time or 0,
            "total_time": total_time,
            "audio_chunks": audio_chunks,
        }

    async def synthesize_full_text(self, text: str) -> dict[str, Any]:
        """Synthesize with full-text mode."""
        self.console.print("[blue]🎤 Synthesizing (full-text)...[/blue]")

        start_time = time.perf_counter()
        audio_chunks = 0

        async for chunk, segment_text in self.tts.stream(text):
            audio_chunks += 1
            await self.play_and_log(chunk, segment_text, audio_chunks)

        total_time = time.perf_counter() - start_time
        return {
            "first_audio_latency": total_time,  # Full-text waits for everything
            "total_time": total_time,
            "audio_chunks": audio_chunks,
        }

    async def synthesize_raw_engine(self, text: str) -> dict[str, Any]:
        """Synthesize with raw engine API."""
        self.console.print("[blue]🎤 Synthesizing (raw engine)...[/blue]")

        start_time = time.perf_counter()

        # Get engine and synthesize directly
        from nanotts.model import manager

        engine = await manager.get(self.tts._model, **self.tts._engine_kwargs)
        chunk = await engine.synth(text)

        total_time = time.perf_counter() - start_time

        if self.verbose:
            self.console.print(
                f'[dim]Raw synthesis: "{text}" → {len(chunk.data)} bytes[/dim]'
            )

        await self.play_and_log(chunk, text, 1)

        return {
            "first_audio_latency": total_time,
            "total_time": total_time,
            "audio_chunks": 1,
        }

    async def run_performance_test(self):
        """Run performance comparison on last response."""
        if not self.last_response:
            self.console.print(
                "[yellow]No previous response to test. Chat first![/yellow]"
            )
            return

        self.console.print(
            Panel(f'Testing: "{self.last_response[:50]}..."', title="Performance Test")
        )

        # Simulate token stream
        async def simulate_token_stream():
            words = self.last_response.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await anyio.sleep(0.01)

        # Test all modes
        results = {}
        results["token-feeding"] = await self.synthesize_streaming(
            simulate_token_stream()
        )
        results["full-text"] = await self.synthesize_full_text(self.last_response)
        results["raw-engine"] = await self.synthesize_raw_engine(self.last_response)

        # Show results
        table = Table(title="Performance Comparison")
        table.add_column("Mode", style="cyan")
        table.add_column("First Audio (ms)", justify="right", style="green")
        table.add_column("Total Time (ms)", justify="right", style="blue")
        table.add_column("Audio Chunks", justify="right", style="yellow")

        for mode, data in results.items():
            table.add_row(
                mode,
                f"{data['first_audio_latency'] * 1000:.1f}",
                f"{data['total_time'] * 1000:.1f}",
                str(data["audio_chunks"]),
            )

        self.console.print(table)

    async def switch_language(self):
        """Switch between English and Chinese voices."""
        # Toggle language
        self.language = "zh" if self.language == "en" else "en"

        try:
            # Reinitialize TTS with new voice
            voice = self.voices[self.language]
            self.tts = NanoTTS(model="edge", voice=voice)

            lang_name = (
                "English (Australian)"
                if self.language == "en"
                else "Chinese (Mainland)"
            )
            self.console.print(
                f"[green]🌐 Switched to {lang_name} voice: {voice}[/green]"
            )

        except (ValueError, ImportError):
            # Fallback to dummy if edge-tts not available
            self.tts = NanoTTS(model="dummy")
            lang_name = "English" if self.language == "en" else "Chinese"
            self.console.print(
                f"[yellow]🌐 Switched to {lang_name} (dummy engine)[/yellow]"
            )

    async def handle_chat(self, user_input: str):
        """Handle regular chat interaction."""
        self.console.print(f"[green]You:[/green] {user_input}")

        response_parts = []

        if self.mode == SynthMode.TOKEN_FEEDING:
            # Token-feeding mode: synthesize as tokens arrive
            async def response_and_collect():
                async for token in self.get_llm_response(user_input):
                    response_parts.append(token)
                    yield token

            perf_data = await self.synthesize_streaming(response_and_collect())

        else:  # full-text mode
            # Collect full response first
            async for token in self.get_llm_response(user_input):
                response_parts.append(token)

            full_response = "".join(response_parts)
            self.console.print(f"[blue]Assistant:[/blue] {full_response}")

            perf_data = await self.synthesize_full_text(full_response)

        self.last_response = "".join(response_parts)

        # Show timing info
        if self.verbose:
            self.console.print(
                f"[dim]⏱️ First audio: {perf_data['first_audio_latency'] * 1000:.1f}ms, "
                f"Total: {perf_data['total_time'] * 1000:.1f}ms[/dim]"
            )

    async def run(self):
        """Main demo loop."""
        if not self.load_config():
            return

        self.console.print(
            Panel("🎤 Interactive TTS Demo", subtitle="Type /help for commands")
        )
        self.show_help()

        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input == "/help":
                    self.show_help()
                elif user_input == "/exit":
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input == "/mode":
                    self.mode = (
                        SynthMode.FULL_TEXT
                        if self.mode == SynthMode.TOKEN_FEEDING
                        else SynthMode.TOKEN_FEEDING
                    )
                    self.console.print(
                        f"[yellow]Switched to {self.mode.value} mode[/yellow]"
                    )
                elif user_input == "/verbose":
                    self.verbose = not self.verbose
                    self.console.print(
                        f"[yellow]Verbose mode: {'ON' if self.verbose else 'OFF'}[/yellow]"
                    )
                elif user_input == "/audio":
                    self.audio_enabled = not self.audio_enabled
                    self.console.print(
                        f"[yellow]Audio playback: {'ON' if self.audio_enabled else 'OFF'}[/yellow]"
                    )
                elif user_input == "/language":
                    await self.switch_language()
                elif user_input == "/performance":
                    await self.run_performance_test()
                else:
                    # Regular chat
                    await self.handle_chat(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


def main():
    """Entry point for the demo."""
    demo = InteractiveDemo()
    asyncio.run(demo.run())


if __name__ == "__main__":
    main()
