"""Build a FireRed VAD session around shared, inference-only model weights."""

from copy import copy
from pathlib import Path


def new_session_vad(template, model_dir):
    from fireredvad import FireRedStreamVad
    from fireredvad.core.audio_feat import AudioFeat
    from fireredvad.core.stream_vad_postprocessor import StreamVadPostprocessor

    config = copy(template.config)
    return FireRedStreamVad(
        AudioFeat(str(Path(model_dir) / "cmvn.ark")), template.vad_model,
        StreamVadPostprocessor(config.smooth_window_size, config.speech_threshold,
                               config.pad_start_frame, config.min_speech_frame,
                               config.max_speech_frame, config.min_silence_frame),
        config,
    )
