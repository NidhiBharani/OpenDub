"""Cheap, dependency-free quality checks shared by pipeline reports and benchmarks.

Optional recognizers/embedders are injected by callers. This module never downloads a model.
A missing measurement has an explicit status so it cannot pass a quality gate by omission.
"""
from __future__ import annotations

import asyncio
import math
import struct
import unicodedata
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def _db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else -math.inf


def _samples(raw: bytes, width: int) -> list[float]:
    if width == 1:
        return [(v - 128) / 128.0 for v in raw]
    if width == 2:
        return [v / 32768.0 for (v,) in struct.iter_unpack('<h', raw)]
    if width == 3:
        return [int.from_bytes(raw[i:i + 3], 'little', signed=True) / 8388608.0
                for i in range(0, len(raw), 3)]
    if width == 4:
        return [v / 2147483648.0 for (v,) in struct.iter_unpack('<i', raw)]
    raise ValueError(f'unsupported PCM width: {width}')


def audio_stats(path: str | Path) -> dict[str, Any]:
    """Inspect PCM WAV audio; missing/unreadable files return a status and reason.

    The peak is a sample peak, not an inter-sample true peak. ``clipping_fraction`` counts
    samples at or above -0.1 dBFS. An empty WAV is readable but has zero duration.
    """
    path = Path(path)
    if not path.exists():
        return {'status': 'missing', 'reason': f'{path} does not exist', 'duration_s': None,
                'peak_dbfs': None, 'rms_dbfs': None, 'clipping_fraction': None}
    try:
        with wave.open(str(path), 'rb') as wav:
            if wav.getcomptype() != 'NONE':
                raise ValueError('compressed WAV is unsupported')
            channels, rate, width, frames = (wav.getnchannels(), wav.getframerate(),
                                             wav.getsampwidth(), wav.getnframes())
            if channels < 1 or rate < 1 or width not in (1, 2, 3, 4):
                raise ValueError('invalid PCM WAV format')
            peak = total_sq = 0.0
            count = clipped = 0
            while True:
                raw = wav.readframes(65536)
                if not raw:
                    break
                for sample in _samples(raw, width):
                    peak = max(peak, abs(sample))
                    total_sq += sample * sample
                    clipped += abs(sample) >= 10 ** (-0.1 / 20)
                    count += 1
            if count != frames * channels:
                raise ValueError('truncated PCM data does not match WAV frame count')
    except (OSError, EOFError, wave.Error, ValueError, struct.error) as exc:
        return {'status': 'unreadable', 'reason': str(exc), 'duration_s': None,
                'peak_dbfs': None, 'rms_dbfs': None, 'clipping_fraction': None}
    return {'status': 'ok', 'reason': '', 'duration_s': frames / rate,
            'peak_dbfs': _db(peak), 'rms_dbfs': _db(math.sqrt(total_sq / count)) if count else -math.inf,
            'clipping_fraction': clipped / count if count else 0.0,
            'channels': channels, 'sample_rate': rate, 'sample_count': count,
            'peak_kind': 'sample'}


def _distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def text_error_rates(reference: str, hypothesis: str) -> dict[str, float | None]:
    """Unicode casefolded WER/CER with punctuation ignored and marks preserved.

    CER excludes whitespace so subtitle spacing cannot dominate character accuracy.
    """
    def normalize(text: str) -> str:
        clean = ''.join(' ' if unicodedata.category(ch).startswith('P') else ch
                        for ch in unicodedata.normalize('NFKC', text).casefold())
        return ' '.join(clean.split())

    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return {'wer': None, 'cer': None}
    rw, hw = ref.split(), hyp.split()
    ref_chars, hyp_chars = list(ref.replace(' ', '')), list(hyp.replace(' ', ''))
    return {'wer': _distance(rw, hw) / len(rw),
            'cer': _distance(ref_chars, hyp_chars) / len(ref_chars)}


def coverage_diagnostics(intervals: Iterable[tuple[float, float]], duration_s: float) -> dict[str, Any]:
    """Union coverage and timing faults for ASR/diarization intervals."""
    source = list(intervals)
    valid = sorted((max(0.0, float(a)), min(float(duration_s), float(b)))
                   for a, b in source if math.isfinite(a) and math.isfinite(b)
                   and b > a and b > 0 and a < duration_s)
    invalid = 0
    # Count invalid original spans independently, including out-of-bounds timestamps.
    for a, b in source:
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration_s):
            invalid += 1
    covered = overlap = 0.0
    overlap_pairs = 0
    cursor = 0.0
    for start, end in valid:
        if start < cursor:
            overlap += min(cursor, end) - start if end > start else 0.0
            overlap_pairs += 1
        covered += max(0.0, end - max(start, cursor))
        cursor = max(cursor, end)
    return {'coverage_ratio': covered / duration_s if duration_s > 0 else None,
            'covered_s': covered, 'overlap_s': overlap, 'overlap_pairs': overlap_pairs,
            'invalid_intervals': invalid, 'segment_count': len(source)}


async def verify_speech(path: str | Path, reference_text: str, model: Any = None,
                        language: str | None = None) -> dict[str, Any]:
    """Roundtrip ASR with an injected local faster-whisper-compatible model.

    No model means 'unavailable'. A model's ``transcribe`` method runs in a worker thread so
    decoding does not block the event loop. The method may return (segments, info) or segments.
    """
    stats = audio_stats(path)
    if stats['status'] != 'ok':
        return {'status': stats['status'], 'reason': stats['reason'], 'wer': None, 'cer': None,
                'transcript': None}
    if model is None:
        return {'status': 'unavailable', 'reason': 'no local ASR verifier configured',
                'wer': None, 'cer': None, 'transcript': None}
    if not reference_text.strip():
        return {'status': 'unavailable', 'reason': 'empty reference text',
                'wer': None, 'cer': None, 'transcript': None}

    def run() -> str:
        method = getattr(model, 'transcribe', model)
        try:
            result = method(str(path), language=language)
        except TypeError:
            result = method(str(path))
        segments = result[0] if isinstance(result, tuple) else result
        return ' '.join(str(getattr(item, 'text', item)).strip() for item in segments).strip()

    try:
        transcript = await asyncio.to_thread(run)
    except Exception as exc:  # noqa: BLE001 - external model may raise arbitrary errors
        return {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}', 'wer': None,
                'cer': None, 'transcript': None}
    return {'status': 'ok', 'reason': '', 'transcript': transcript,
            **text_error_rates(reference_text, transcript)}


async def speaker_similarity(reference_path: str | Path, take_path: str | Path,
                             embedder: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Cosine similarity from an injected local embedding callable, with honest missing state."""
    for path in (reference_path, take_path):
        stats = audio_stats(path)
        if stats['status'] != 'ok':
            return {'status': stats['status'], 'reason': stats['reason'], 'cosine': None}
    if embedder is None:
        return {'status': 'unavailable', 'reason': 'no local speaker embedder configured',
                'cosine': None}
    try:
        a, b = await asyncio.gather(asyncio.to_thread(embedder, str(reference_path)),
                                    asyncio.to_thread(embedder, str(take_path)))
        a, b = list(a), list(b)
        if len(a) != len(b) or not a:
            raise ValueError('embedding dimensions differ or are empty')
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm = math.sqrt(sum(float(x) ** 2 for x in a) * sum(float(y) ** 2 for y in b))
        if not math.isfinite(dot) or not math.isfinite(norm):
            raise ValueError("nonfinite embedding")
        if not norm:
            raise ValueError('zero-norm embedding')
        return {'status': 'ok', 'reason': '', 'cosine': max(-1.0, min(1.0, dot / norm))}
    except Exception as exc:  # noqa: BLE001 - external model may raise arbitrary errors
        return {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}', 'cosine': None}


def load_speaker_embedder(local_path: str | Path) -> Callable[[str], Any]:
    """Load SpeechBrain ECAPA weights from an existing local model directory only.

    Callers explicitly configure the directory and inject the returned callable into
    :func:`speaker_similarity`. Absence of a model or package raises an error; no Hub
    fallback is attempted. The directory must contain SpeechBrain's hyperparams.yaml.
    """
    path = Path(local_path).resolve()
    if not path.is_dir() or not (path / 'hyperparams.yaml').is_file():
        raise FileNotFoundError(f'local SpeechBrain model missing hyperparams.yaml: {path}')
    from speechbrain.inference.speaker import EncoderClassifier  # lazy optional dependency
    from speechbrain.utils.fetching import FetchConfig
    classifier = EncoderClassifier.from_hparams(
        source=str(path), savedir=str(path), fetch_config=FetchConfig(allow_network=False))

    def embed(audio_path: str) -> Any:
        signal = classifier.load_audio(audio_path)
        return classifier.encode_batch(signal.unsqueeze(0)).detach().cpu().flatten().tolist()

    return embed


def window_rms(path: str | Path, start_s: float, end_s: float) -> float | None:
    """Channel-energy RMS amplitude in a PCM WAV window, or None if unreadable/empty."""
    if end_s <= start_s:
        return None
    try:
        with wave.open(str(path), 'rb') as wav:
            rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
            wav.setpos(min(max(0, int(start_s * rate)), wav.getnframes()))
            values = _samples(wav.readframes(int((end_s - start_s) * rate)), width)
    except (OSError, wave.Error, ValueError):
        return None
    if not values or channels < 1:
        return None
    # Measure channel energy, not a mono downmix that cancels anti-phase stereo music.
    return math.sqrt(sum(value * value for value in values) / len(values))


async def mix_statistics(
    mix_path: str | Path, vocal_path: str | Path, background_path: str | Path,
    intervals: Iterable[tuple[float, float]], *, gains: dict[str, float] | None = None,
    target_lufs: float = -16.0,
) -> dict[str, Any]:
    """Measure a mastered mix and approximate stem balance in the same speech windows.

    Stem margin uses the reported static gains. Limiting and optional ducking change the
    realized contribution, so this is a provisional diagnostic, not a proof of audibility.
    All unavailable measurements retain an explicit status/reason.
    """
    from ..media import ffmpeg

    mix_path = Path(mix_path)
    stats = await asyncio.to_thread(audio_stats, mix_path)
    if stats['status'] != 'ok':
        return {'status': stats['status'], 'reason': stats['reason'],
                'integrated_lufs': None, 'true_peak_dbtp': None,
                'estimated_stem_margin_db': None}
    result: dict[str, Any] = {
        'status': 'measured', 'sample_peak_dbfs': stats['peak_dbfs'],
        'clipping_fraction': stats['clipping_fraction'], 'target_lufs': target_lufs,
        'integrated_lufs': None, 'true_peak_dbtp': None, 'loudness_error_lu': None,
        'estimated_stem_margin_db': None, 'stem_margin_status': 'unavailable',
        'stem_margin_reason': '',
    }
    try:
        lufs = await ffmpeg.measure_loudness(mix_path)
        result.update(integrated_lufs=lufs, loudness_error_lu=abs(lufs - target_lufs))
    except Exception as exc:  # noqa: BLE001 - failure should not erase other measurements
        result['loudness_reason'] = f'{type(exc).__name__}: {exc}'
    try:
        result['true_peak_dbtp'] = await ffmpeg.measure_true_peak(mix_path)
    except Exception as exc:  # noqa: BLE001 - ffmpeg can fail independently per meter
        result['true_peak_reason'] = f'{type(exc).__name__}: {exc}'

    vocal_gain = (gains or {}).get('vocal_gain_db')
    background_gain = (gains or {}).get('background_gain_db')
    if vocal_gain is None or background_gain is None:
        result['stem_margin_reason'] = 'applied stem gains unavailable'
        return result
    margins = []
    for start, end in intervals:
        vocal, bed = await asyncio.gather(
            asyncio.to_thread(window_rms, vocal_path, start, end),
            asyncio.to_thread(window_rms, background_path, start, end))
        if vocal is None or bed is None or vocal <= 0 or bed <= 0:
            continue
        margins.append(20 * math.log10(vocal / bed) + vocal_gain - background_gain)
    if margins:
        result['estimated_stem_margin_db'] = sum(margins) / len(margins)
        result['stem_margin_status'] = 'estimated'
        result['stem_margin_reason'] = ('static-gain estimate; limiter, ducking and original '
                                        'splices can change the realized balance')
    else:
        result['stem_margin_reason'] = 'no comparable vocal/background samples in speech windows'
    result['stem_windows_measured'] = len(margins)
    return result
