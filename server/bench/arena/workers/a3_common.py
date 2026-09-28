"""Shared helpers for A3/A8 region workers (not a worker): frame posteriors → segments, and
the AudioSet label groups used for speech / music / singing and nonverbal events."""
from __future__ import annotations

SPEECH = {"Speech", "Male speech, man speaking", "Female speech, woman speaking",
          "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling",
          "Whispering", "Speech synthesizer", "Shout", "Yell"}
SINGING = {"Singing", "Choir", "Yodeling", "Chant", "Mantra", "Male singing", "Female singing",
           "Child singing", "Synthetic singing", "Rapping", "Humming", "Vocal music",
           "A capella", "Lullaby"}
MUSIC = {"Music", "Musical instrument", "Background music", "Theme music", "Soundtrack music",
         "Pop music", "Orchestra", "Piano", "Guitar", "Drum", "Electronic music"}
NONVERBAL = {"Laughter": "laughter", "Baby laughter": "laughter", "Giggle": "laughter",
             "Snicker": "laughter", "Belly laugh": "laughter", "Chuckle, chortle": "laughter",
             "Crying, sobbing": "crying", "Baby cry, infant cry": "crying", "Whimper": "crying",
             "Wail, moan": "crying", "Sigh": "sigh", "Screaming": "scream", "Breathing": "breath",
             "Gasp": "gasp", "Pant": "breath", "Cough": "cough", "Sniff": "sniff",
             "Groan": "groan", "Yawn": "yawn", "Throat clearing": "throat_clear"}


def group_of(label: str, task: str = "regions") -> str | None:
    if task == "events":
        return NONVERBAL.get(label)
    if label in SINGING:
        return "singing"
    if label in SPEECH:
        return "speech"
    if label in MUSIC:
        return "music"
    return None


def to_segments(times: list[float], hop: float, probs: dict[str, list[float]],
                threshold: float = 0.5, min_dur: float = 0.2) -> list[dict]:
    """Per-class frame probabilities (frame k centred at ``times[k]``) → merged segments with
    the mean probability as score."""
    segs = []
    for label, ps in probs.items():
        start, acc = None, []
        for t, pr in zip([*times, None], [*ps, 0.0]):
            if t is not None and pr >= threshold:
                if start is None:
                    start, acc = max(0.0, t - hop / 2), []
                acc.append(pr)
            elif start is not None:
                end = (t - hop / 2) if t is not None else times[-1] + hop / 2
                if end - start >= min_dur:
                    segs.append({"start": round(start, 3), "end": round(end, 3), "label": label,
                                 "score": round(sum(acc) / len(acc), 4)})
                start = None
    return sorted(segs, key=lambda s: (s["start"], s["label"]))
