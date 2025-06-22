# nanoTTS

> **Minimalist Text-to-Speech for Python**  
> Zero-config streaming TTS with async support

[![CI](https://github.com/3rd-Musketeer/nanoTTS/actions/workflows/ci.yml/badge.svg)](https://github.com/3rd-Musketeer/nanoTTS/actions)
[![Coverage](https://codecov.io/gh/3rd-Musketeer/nanoTTS/branch/main/graph/badge.svg)](https://codecov.io/gh/3rd-Musketeer/nanoTTS)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## What is nanoTTS?

nanoTTS is a **tiny, powerful TTS library** that just works. No complex configs, no heavyweight dependencies—just clean async streaming with tiktoken-based intelligent segmentation.

```python
from nanotts import NanoTTS

tts = NanoTTS()
async for audio, text in tts.stream("Hello world!"):
    # audio is raw bytes, text is the segment
    print(f"Speaking: {text}")
```

### Why nano?

- **Async-first**: Built for modern Python with `anyio`
- **Streaming**: Real-time synthesis as text flows in
- **Pluggable**: Swap TTS engines without code changes  
- **Lightweight**: Minimal dependencies, maximum performance
- **Zero-config**: Works out of the box with sensible defaults

---

## Quick Start

### Installation

```bash
# Basic installation
pip install nanotts

# With Edge TTS support (Microsoft's free voices)
pip install nanotts[edge]

# Development setup with uv (recommended)
uv add nanotts
```

### Copy-Paste Examples

#### 1. **Basic Streaming**
```python
import asyncio
from nanotts import NanoTTS

async def basic_demo():
    tts = NanoTTS(model="dummy")  # Use dummy for testing
    
    async for audio_chunk, text_segment in tts.stream("Hello! How are you today?"):
        print(f"Generated audio for: '{text_segment}'")
        print(f"Audio size: {len(audio_chunk.data)} bytes")

# Run it!
asyncio.run(basic_demo())
```

#### 2. **Real TTS with Edge**
```python
import asyncio
from nanotts import NanoTTS

async def edge_demo():
    # Uses Microsoft Edge TTS (free, no API key needed!)
    tts = NanoTTS(model="edge", voice="en-US-AriaNeural")
    
    text = "Welcome to nanoTTS! This is streaming text-to-speech."
    
    audio_chunks = []
    async for chunk, segment in tts.stream(text):
        print(f"Synthesized: {segment}")
        audio_chunks.append(chunk.data)
    
    # Save complete audio
    with open("output.mp3", "wb") as f:
        f.write(b"".join(audio_chunks))
    
    print("Saved to output.mp3")

# Run it!
asyncio.run(edge_demo())
```

#### 3. **Smart Segmentation**
```python
import asyncio
from nanotts import NanoTTS

async def segmentation_demo():
    tts = NanoTTS(model="dummy", min_tokens=5, max_tokens=20)  # Force short segments
    
    long_text = """
    This is a long text that will be automatically segmented using tiktoken. 
    The system intelligently breaks at sentence boundaries first, then commas, 
    respecting punctuation and word boundaries for natural speech flow.
    """
    
    segments = []
    async for chunk, text in tts.stream(long_text):
        segments.append(text.strip())
        print(f"Segment {len(segments)}: '{text.strip()}'")
    
    print(f"Total segments: {len(segments)}")

# Run it!
asyncio.run(segmentation_demo())
```

#### 4. **Live Token Streaming**
```python
import asyncio
from nanotts import NanoTTS

async def token_streaming_demo():
    """Simulate LLM token-by-token generation"""
    tts = NanoTTS(model="dummy")
    
    async def llm_tokens():
        # Simulate streaming tokens from ChatGPT/Claude
        tokens = ["Hello", " there", "!", " How", " can", " I", " help", " you", " today", "?"]
        for token in tokens:
            yield token
            await asyncio.sleep(0.1)  # Simulate network delay
    
    print("Streaming tokens into TTS...")
    async for audio, text in tts.stream(llm_tokens()):
        print(f"Audio ready for: '{text}'")

# Run it!
asyncio.run(token_streaming_demo())
```

#### 5. **Multiple Voices & Languages**
```python
import asyncio
from nanotts import NanoTTS

async def multilingual_demo():
    voices = [
        ("en-US-AriaNeural", "Hello from America!"),
        ("en-GB-SoniaNeural", "Greetings from Britain!"), 
        ("fr-FR-DeniseNeural", "Bonjour de France!"),
        ("ja-JP-NanamiNeural", "こんにちは日本から！"),
    ]
    
    for voice, text in voices:
        print(f"Switching to {voice}")
        tts = NanoTTS(model="edge", voice=voice)
        
        async for chunk, segment in tts.stream(text):
            print(f"  {segment}")
        
        print()

# Run it!  
asyncio.run(multilingual_demo())
```

---

## Interactive Demo

Want to try the full experience? Run our interactive demo:

```bash
# Clone and run the demo
git clone https://github.com/3rd-Musketeer/nanoTTS
cd nanoTTS
uv run demo/interactive_demo.py
```

<details>
<summary><strong>Minimal Interactive Demo</strong> (click to expand)</summary>

```python
import asyncio
from nanotts import NanoTTS

async def mini_demo():
    tts = NanoTTS(model="edge")  # or "dummy" for testing
    
    print("nanoTTS Mini Demo")
    print("Type text to hear it speak (or 'quit' to exit):")
    
    while True:
        text = input("\nText: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
            
        if not text:
            continue
            
        print("Synthesizing...")
        audio_data = b""
        
        async for chunk, segment in tts.stream(text):
            print(f"  {segment}")
            audio_data += chunk.data
        
        print(f"Generated {len(audio_data)} bytes of audio!")

if __name__ == "__main__":
    asyncio.run(mini_demo())
```
</details>

---

## Architecture

nanoTTS follows the **nano philosophy**: 
- **Simple**: One main class, clear API
- **Composable**: Pluggable engines and segmentation
- **Async-native**: Built for modern concurrent Python

```
nanotts/
├── NanoTTS          # Main orchestrator  
├── Engine           # TTS plugin protocol
├── Segmenter        # Smart text splitting
├── AudioData        # Audio format handling
└── ModelManager     # Engine registration
```

### Supported Engines

| Engine | Description | Setup |
|--------|-------------|-------|
| `dummy` | Testing & development | Built-in |
| `edge` | Microsoft Edge TTS | `pip install nanotts[edge]` |
| `openai` | OpenAI TTS API | API key required |
| *custom* | Your own engine | Implement `Engine` protocol |

---

## Advanced Usage

### Custom Audio Formats

```python
from nanotts import NanoTTS, AudioSpec

# High-quality 48kHz output
tts = NanoTTS(
    model="edge",
    output_format=AudioSpec("pcm", 48000, 2, 16)  # stereo 16-bit
)
```

### Custom Engine

```python
from nanotts import Engine, AudioChunk, AudioSpec
from nanotts.model import manager

class MyEngine(Engine):
    async def synth(self, text: str, *, target: AudioSpec | None = None) -> AudioChunk:
        # Your TTS logic here
        audio_data = my_tts_function(text)
        return AudioChunk(audio_data, self.output_spec)

# Register it
async def build_my_engine(**kwargs):
    return MyEngine()

manager.register("my-engine", build_my_engine, "My custom TTS engine")
```

### Smart Segmentation Control

```python
tts = NanoTTS(
    model="edge",
    min_tokens=10,     # Minimum tokens per segment
    max_tokens=50,     # Maximum tokens per segment
    timeout_ms=1000,   # Max wait time for more text
)

# Tiered punctuation breaking: sentences first, then commas
async for chunk, text in tts.stream("Dr. Smith has a Ph.D. However, students often struggle, but eventually succeed!"):
    # Breaks at strong punctuation (.) first, commas as fallback
    print(f"Segment: {text}")
```

---

## Testing & Development

```bash
# Development setup (one-time)
uv sync --group dev
uv run pre-commit install  # Auto-format on every commit

# Run tests
uv run pytest

# Check coverage  
uv run pytest --cov=nanotts

# Manual formatting (optional - pre-commit handles this)
uv run ruff check . --fix
uv run ruff format .
```

---

## API Reference

### `NanoTTS`

```python
NanoTTS(
    model: str = "dummy",           # Engine to use
    min_tokens: int = 10,           # Minimum tokens per segment
    max_tokens: int = 50,           # Maximum tokens per segment  
    timeout_ms: int = 800,          # Timeout for streaming
    output_format: AudioSpec = ..., # Audio format
    **engine_kwargs                 # Engine-specific options
)
```

**Methods:**
- `async stream(text) -> AsyncIterator[AudioChunk, str]` - Main streaming interface
- `cancel()` - Stop current synthesis

### `AudioChunk` & `AudioSpec`

```python
@dataclass
class AudioSpec:
    codec: str              # "pcm", "mp3", "opus"
    sample_rate: int        # 16000, 24000, 48000
    channels: int           # 1 (mono), 2 (stereo)  
    sample_width: int | None # 16, 24, 32 (for PCM)

@dataclass  
class AudioChunk:
    data: bytes            # Raw audio data
    spec: AudioSpec        # Format information
```

---

## Contributing

We love contributions! nanoTTS follows the **nano philosophy**:

1. **Keep it simple** - No unnecessary complexity
2. **Stay lightweight** - Minimal dependencies  
3. **Test everything** - 85%+ coverage required
4. **Document clearly** - Code should be self-explaining
5. **Automate the boring stuff** - Pre-commit hooks handle formatting automatically

```bash
# Development setup
git clone https://github.com/3rd-Musketeer/nanoTTS
cd nanoTTS
uv sync --group dev

# Run the full test suite
uv run pytest

# Check your changes
uv run ruff check .
uv run ruff format .
```

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## That's It!

nanoTTS: **Tiny library, huge possibilities**

Questions? Issues? [Open a GitHub issue](https://github.com/3rd-Musketeer/nanoTTS/issues) or start a [discussion](https://github.com/3rd-Musketeer/nanoTTS/discussions)!