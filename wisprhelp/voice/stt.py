"""Whisper Large v3 Multilingual Speech-to-Text Transcriber.

Uses Groq API for sub-second cloud transcription, with clean fallback
for offline testing and development. Supports English, Hindi, and Hinglish.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Optional

from wisprhelp.config import config

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Production Speech-To-Text client for Whisper Large v3."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3",
        language_prompt: str = "English, Hindi, Hinglish, bilingual conversation",
    ) -> None:
        self.api_key = api_key or config.groq_api_key
        self.model = model
        self.language_prompt = language_prompt
        self._groq_client = None

        if self.api_key:
            try:
                from groq import Groq

                self._groq_client = Groq(api_key=self.api_key)
                logger.info("Initialized Groq Whisper Large v3 client.")
            except ImportError:
                logger.warning("groq library not found. Will use mock transcription fallback.")
        else:
            logger.info("No GROQ_API_KEY configured. Running in offline fallback mode.")

    @staticmethod
    def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
        """Converts raw PCM audio bytes to in-memory WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

    @staticmethod
    def _compute_rms(pcm_bytes: bytes) -> float:
        """Computes approximate RMS amplitude without external dependencies (Python 3.14 compatible)."""
        import math
        import struct

        count = len(pcm_bytes) // 2
        if count == 0:
            return 0.0
        step = max(1, count // 1000)
        sum_sq = 0
        samples = 0
        for i in range(0, len(pcm_bytes) - 1, 2 * step):
            val = struct.unpack_from("<h", pcm_bytes, i)[0]
            sum_sq += val * val
            samples += 1
        if samples == 0:
            return 0.0
        return math.sqrt(sum_sq / samples)

    @classmethod
    def sanitize_transcription(cls, raw_text: str) -> str:
        """Sanitizes raw Whisper output, filtering prompt echoes and hallucinations."""
        import re

        if not raw_text:
            return ""

        text = raw_text.strip().strip("\"'.,!?:; ")
        if not text:
            return ""

        lower = text.lower()

        # Known phantom subtitle substrings
        hallucination_substrings = [
            "subtitles by",
            "subtitle by",
            "amara.org",
            "thank you for watching",
            "thanks for watching",
            "subscribe to my channel",
            "please subscribe",
            "bilingual conversation",
        ]
        if any(h_sub in lower for h_sub in hallucination_substrings):
            logger.debug("Filtered out Whisper subtitle hallucination: '%s'", text)
            return ""

        # Exact match hallucinations
        exact_hallucinations = {
            "thank you",
            "thank you.",
            "thank you very much",
            "you",
            "you.",
            "bye",
            "bye.",
        }
        if lower in exact_hallucinations:
            logger.debug("Filtered out exact Whisper hallucination: '%s'", text)
            return ""

        # Check if text only consists of words from the prompt
        prompt_words = {"english", "hindi", "hinglish", "bilingual", "conversation"}
        tokens = set(re.findall(r"\w+", lower))
        if tokens and tokens.issubset(prompt_words):
            logger.debug("Filtered out Whisper prompt echo: '%s'", text)
            return ""

        # Phonetic normalization for programming extensions (e.g. 'main.by' -> 'main.py')
        text = re.sub(r"\b([a-zA-Z0-9_\-]+)\.(by|pie)\b", r"\1.py", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([a-zA-Z0-9_\-]+)\s+dot\s+(by|pie|py)\b", r"\1.py", text, flags=re.IGNORECASE)

        return text

    def transcribe(self, pcm_audio: bytes, sample_rate: int = 16000) -> str:
        """Transcribes PCM audio bytes into text.

        Args:
            pcm_audio: Raw 16-bit PCM mono audio bytes.
            sample_rate: Sample rate in Hz (default 16000).

        Returns:
            str: Transcribed text query.
        """
        if not pcm_audio:
            return ""

        # Check if audio has sufficient energy above ambient background noise
        rms = self._compute_rms(pcm_audio)
        if rms < 70.0:
            logger.debug("Audio energy too low for STT (RMS=%.1f < 70). Skipping.", rms)
            return ""

        wav_bytes = self.pcm_to_wav_bytes(pcm_audio, sample_rate=sample_rate)

        # 1. Primary path: Groq API
        if self._groq_client is not None:
            try:
                wav_file_tuple = ("audio.wav", wav_bytes, "audio/wav")
                transcription = self._groq_client.audio.transcriptions.create(
                    file=wav_file_tuple,
                    model=self.model,
                    prompt=self.language_prompt,
                    temperature=0.0,
                    response_format="json",
                )
                raw_text = transcription.text.strip()
                cleaned = self.sanitize_transcription(raw_text)
                if cleaned:
                    logger.info("Groq STT transcription: '%s'", cleaned)
                else:
                    logger.info("Groq STT returned filtered/empty text (raw: '%s')", raw_text)
                return cleaned
            except Exception as exc:
                logger.error("Groq Whisper API call failed: %s. Reverting to fallback.", exc)

        # 2. Secondary fallback: Local or mock transcription
        return self._offline_transcribe(pcm_audio)

    def _offline_transcribe(self, pcm_audio: bytes) -> str:
        """Fallback transcription when Groq API is unavailable."""
        logger.debug("Executing offline speech fallback for %d audio bytes.", len(pcm_audio))
        # Deterministic simulation or local engine
        return ""
