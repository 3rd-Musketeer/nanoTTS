# Interactive TTS Demo

A comprehensive demo showcasing nanoTTS with LLM integration and performance comparison.

## Setup

1. **Install dependencies:**
   ```bash
   uv sync --group dev
   ```

2. **Configure API access:**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

3. **Run the demo:**
   ```bash
   uv run python demo/interactive_demo.py
   ```

## Features

### Synthesis Modes
- **Token-feeding**: Real-time synthesis as LLM tokens arrive (lower latency)
- **Full-text**: Wait for complete response, then synthesize (higher quality)

### Commands
- `/mode` - Switch between token-feeding and full-text modes
- `/verbose` - Toggle detailed segmentation output
- `/performance` - Benchmark last response across all synthesis methods
- `/help` - Show command reference
- `/exit` - Quit demo

### Performance Testing

The `/performance` command tests three synthesis approaches:
1. **Token-feeding**: Simulates real-time token streaming
2. **Full-text**: Complete text synthesis
3. **Raw engine**: Direct engine API call

Metrics displayed:
- First audio chunk latency
- Total synthesis time  
- Number of audio segments

## Example Session

```
> Hello, tell me about Python
Assistant: Python is a high-level programming language...
[Synthesizing in real-time]

> /performance
Testing: "Python is a high-level programming language..."
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Mode         ┃ First Audio (ms) ┃ Total Time (ms) ┃ Audio Chunks ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ token-feeding│ 45.2            │ 1250.8         │ 8           │
│ full-text    │ 1180.4          │ 1180.4         │ 8           │  
│ raw-engine   │ 890.1           │ 890.1          │ 1           │
└──────────────┴──────────────────┴─────────────────┴──────────────┘

> /mode
Switched to full-text mode

> /exit
Goodbye!
```

## Configuration

Edit `.env` file:
```env
API_KEY=your_api_key_here
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o-mini
```

Works with any OpenAI-compatible API (OpenAI, Anthropic, local models, etc.).