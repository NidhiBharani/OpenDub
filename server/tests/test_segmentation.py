"""C1 · pause + punctuation line segmentation over word timing."""
from app.models import ASRSegment, Word
from app.pipeline.segmentation import resegment


def _words(spec: list[tuple[float, float, str]]) -> list[Word]:
    return [Word(start=s, end=e, text=t) for s, e, t in spec]


def test_splits_on_pause_and_sentence_end_and_merges_fragments():
    # The recognizer glued two sentences into one segment and cut a third across two.
    a = ASRSegment(start=0, end=4, text="x", words=_words([
        (0.0, 0.4, " Hello"), (0.45, 0.9, " there."), (1.1, 1.5, " How"), (1.55, 2.0, " are"),
    ]))
    b = ASRSegment(start=2, end=5, text="y", words=_words([
        (2.05, 2.4, " you"), (2.45, 2.9, " today?"), (4.0, 4.5, " Fine"),
    ]))
    lines = resegment([a, b])
    assert [l.text for l in lines] == ["Hello there.", "How are you today?", "Fine"]
    assert lines[1].start == 1.1 and lines[1].end == 2.9
    assert [w.text for w in lines[2].words] == [" Fine"]


def test_unspaced_language_joins_without_spaces():
    seg = ASRSegment(start=0, end=2, text="", words=_words([(0, 0.5, "こん"), (0.5, 1.0, "にちは。")]))
    assert resegment([seg])[0].text == "こんにちは。"


def test_long_run_is_capped_at_its_widest_gap():
    spec = [(i * 1.0, i * 1.0 + 0.9, f" w{i}") for i in range(20)]
    spec[10] = (10.25, 10.9, " w10")  # widest internal gap (still under min_pause)
    lines = resegment([ASRSegment(start=0, end=20, text="", words=_words(spec))])
    assert len(lines) == 2 and lines[1].words[0].text == " w10"
    assert all(l.end - l.start <= 12 for l in lines)


def test_segments_without_words_pass_through():
    seg = ASRSegment(start=1, end=2, text="[line 1]")
    assert resegment([seg]) == [seg]
