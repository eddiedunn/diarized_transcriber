"""Audio loading and processing utilities."""

import logging
import tempfile
from pathlib import Path
from typing import Union, BinaryIO
import soundfile as sf
import whisperx
from pydantic import HttpUrl
from ..exceptions import AudioProcessingError
from ..models.content import MediaContent
from ..types import AudioArray

logger = logging.getLogger(__name__)

def load_audio(
    source: Union[str, Path, HttpUrl, BinaryIO],
    sample_rate: int = 16000
) -> tuple[AudioArray, int]:
    """
    Load audio from various sources and ensure consistent format.
    
    Args:
        source: Audio source - can be a URL, file path, or file-like object
        sample_rate: Target sample rate for the audio
        
    Returns:
        tuple[AudioArray, int]: Audio array and sample rate
        
    Raises:
        AudioProcessingError: If audio loading or processing fails
    """
    logger.debug("Loading audio from: %s", source)
    
    try:
        if isinstance(source, HttpUrl):
            # Use whisperx's built-in audio loading for URLs
            audio = whisperx.load_audio(str(source))
            return audio, sample_rate

        # For file paths and file-like objects, use soundfile
        # dtype="float32" avoids float64 default which causes dtype
        # mismatches in pyannote's batch norm layers on CUDA
        audio_data, file_sample_rate = sf.read(source, dtype="float32")
        
        # Resample if necessary
        if file_sample_rate != sample_rate:
            logger.debug(
                "Resampling audio from %dHz to %dHz",
                file_sample_rate,
                sample_rate,
            )

            import torch
            import torchaudio

            waveform = torch.from_numpy(audio_data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            resampler = torchaudio.transforms.Resample(
                orig_freq=file_sample_rate,
                new_freq=sample_rate,
            )
            waveform = resampler(waveform)

            if waveform.shape[0] == 1:
                audio_data = waveform.squeeze(0).numpy()
            else:
                audio_data = waveform.transpose(0, 1).numpy()

        return audio_data, sample_rate
            
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {str(e)}") from e

def create_temp_audio_file(
    audio_data: AudioArray,
    sample_rate: int
) -> tuple[str, tempfile.NamedTemporaryFile]:
    """
    Create a temporary WAV file from audio data.
    
    Args:
        audio_data: Audio array
        sample_rate: Sample rate of the audio
        
    Returns:
        tuple[str, NamedTemporaryFile]: Path to temp file and file object
        
    Raises:
        AudioProcessingError: If creating temp file fails
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_data, sample_rate)
        return temp_file.name, temp_file
    except Exception as e:
        raise AudioProcessingError(
            f"Failed to create temporary audio file: {str(e)}"
        ) from e

def process_media_content(content: MediaContent) -> tuple[AudioArray, int]:
    """
    Process audio from a MediaContent object.
    
    Args:
        content: MediaContent object containing media information
        
    Returns:
        tuple[AudioArray, int]: Processed audio array and sample rate
        
    Raises:
        AudioProcessingError: If processing fails
    """
    logger.info("Processing audio for content: %s", content.title)
    return load_audio(content.media_url)
