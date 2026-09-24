"""C1 · dub-line segmentation — regroup recognized words into speakable lines.

The recognizer's own segments follow its decoding windows, not the performance: they glue two
sentences across a breath, or cut one mid-phrase. With word timing we can do better without any
model: split where the speaker actually pauses or a sentence ends, and cap the line length so a
voice model never gets a 25-second paragraph.
"""
from __future__ import annotations

from ..models import ASRSegment, Word

_SENTENCE_END = tuple("。．.!?！？…‥")
_CLOSERS = "\"'”’」』）)]"
_SENTENCE_GAP = 0.12  # a sentence end only splits when followed by at least this much air (s)


def _ends_sentence(word: Word) -> bool:
    return word.text.strip().rstrip(_CLOSERS).endswith(_SENTENCE_END)


def _line(words: list[Word]) -> ASRSegment:
    text = "".join(w.text for w in words).strip()
    return ASRSegment(start=words[0].start, end=words[-1].end, text=text, words=words)


def _cap_length(words: list[Word], max_seconds: float) -> list[list[Word]]:
    """Split an over-long run at its widest internal gap, recursively."""
    if len(words) < 2 or words[-1].end - words[0].start <= max_seconds:
        return [words]
    # Ignore the outer 15 % so we never shave a single word off an end.
    lo = max(1, int(len(words) * 0.15))
    hi = min(len(words) - 1, len(words) - lo)
    candidates = range(lo, hi + 1) if lo <= hi else range(1, len(words))
    cut = max(candidates, key=lambda i: words[i].start - words[i - 1].end)
    return _cap_length(words[:cut], max_seconds) + _cap_length(words[cut:], max_seconds)


def resegment(
    segments: list[ASRSegment], min_pause: float = 0.35, max_line_seconds: float = 12.0
) -> list[ASRSegment]:
    """Lines rebuilt from word timing. Segments without words pass through untouched, and act
    as hard boundaries (we cannot know where inside them a pause falls)."""
    out: list[ASRSegment] = []
    run: list[Word] = []

    def flush() -> None:
        if run:
            out.extend(_line(chunk) for chunk in _cap_length(list(run), max_line_seconds))
            run.clear()

    for seg in segments:
        words = [w for w in seg.words if w.text.strip() and w.end >= w.start]
        if not words:
            flush()
            out.append(seg)
            continue
        for word in words:
            if run:
                gap = word.start - run[-1].end
                if gap >= min_pause or (_ends_sentence(run[-1]) and gap >= _SENTENCE_GAP):
                    flush()
            run.append(word)
    flush()
    return sorted(out, key=lambda s: (s.start, s.end))
