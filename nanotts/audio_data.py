from __future__ import annotations

from dataclasses import dataclass
import shutil

import anyio


@dataclass(frozen=True)
class AudioSpec:
    """Audio format specification."""

    codec: str
    sample_rate: int
    channels: int
    sample_width: int | None = None


@dataclass
class AudioChunk:
    """Audio data with format specification."""

    data: bytes
    spec: AudioSpec


class UnsupportedFormat(Exception):
    """Raised when audio format conversion is not supported."""

    def __init__(self, source: AudioSpec, target: AudioSpec):
        self.source = source
        self.target = target
        super().__init__(f"Cannot convert from {source} to {target}")


class AudioTranscoder:
    """Handles audio format conversion using ffmpeg."""

    @staticmethod
    async def convert(chunk: AudioChunk, target: AudioSpec) -> AudioChunk:
        """Convert audio chunk to target format."""
        if chunk.spec == target:
            return chunk

        # Try ffmpeg-python first, then fall back to manual ffmpeg
        try:
            import ffmpeg

            def do_transcode(
                input_data: bytes, source_spec: AudioSpec, target_spec: AudioSpec
            ) -> bytes:
                """Synchronous transcoding using ffmpeg-python"""
                # Determine input format
                input_format = source_spec.codec
                if input_format == "pcm":
                    input_format = f"s{source_spec.sample_width}le"

                # Determine output format and codec
                if target_spec.codec == "pcm":
                    output_format = "wav"
                    output_codec = f"pcm_s{target_spec.sample_width}le"
                else:
                    output_format = target_spec.codec
                    output_codec = None

                # Build ffmpeg pipeline
                process = (
                    ffmpeg.input(
                        "pipe:",
                        format=input_format,
                        ar=source_spec.sample_rate,
                        ac=source_spec.channels,
                    )
                    .output(
                        "pipe:",
                        format=output_format,
                        acodec=output_codec,
                        ar=target_spec.sample_rate,
                        ac=target_spec.channels,
                    )
                    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                )

                stdout, stderr = process.communicate(input=input_data)

                if process.returncode != 0:
                    raise RuntimeError(f"FFmpeg transcoding failed: {stderr.decode()}")

                return stdout

            # Run transcoding in thread pool
            result = await anyio.to_thread.run_sync(
                do_transcode, chunk.data, chunk.spec, target
            )
            return AudioChunk(result, target)

        except ImportError:
            # ffmpeg-python not available, fall back to manual approach
            pass
        except Exception:
            # Transcoding failed, try manual approach
            pass

        # Fallback: manual ffmpeg (original implementation)
        if AudioTranscoder.is_ffmpeg_available():
            input_format = AudioTranscoder.get_ffmpeg_format(chunk.spec)
            output_format = AudioTranscoder.get_ffmpeg_format(target)

            if input_format and output_format:
                try:
                    import subprocess

                    process = await anyio.open_process(
                        [
                            "ffmpeg",
                            "-f",
                            input_format,
                            "-i",
                            "pipe:0",
                            "-f",
                            output_format,
                            "-ar",
                            str(target.sample_rate),
                            "-ac",
                            str(target.channels),
                            "pipe:1",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )

                    await process.stdin.send(chunk.data)
                    await process.stdin.aclose()

                    # Read all output from ffmpeg
                    result = b""
                    async for data in process.stdout:
                        result += data

                    await process.wait()

                    return AudioChunk(result, target)
                except Exception:
                    pass

        raise UnsupportedFormat(chunk.spec, target)

    @staticmethod
    def is_ffmpeg_available() -> bool:
        """Check if ffmpeg binary is available."""
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def get_ffmpeg_format(spec: AudioSpec) -> str | None:
        """Get ffmpeg format string for AudioSpec."""
        if spec.codec == "pcm":
            if spec.sample_width == 16:
                return "s16le"
            elif spec.sample_width == 24:
                return "s24le"
            elif spec.sample_width == 32:
                return "s32le"
        elif spec.codec == "mp3":
            return "mp3"
        elif spec.codec == "opus":
            return "opus"
        return None
