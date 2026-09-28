"""Quality gates exercise stage orchestration, retries, and reference isolation."""
from pathlib import Path

import pytest

from app import config, store
from app.media import ffmpeg
from app.models import Job, MediaInfo, Project, Segment, Speaker, TTSRequest
from app.pipeline import runtime_quality as rq
from app.pipeline import stages
from app.providers.base import ProviderMeta, TTSProvider


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'PROJECTS_DIR', tmp_path / 'projects')
    monkeypatch.setattr(rq, 'options', lambda: {'emotion_model': 'off', 'tts_attempts': 2})
    return Project(name='quality', source_lang='ja', media=MediaInfo(duration=4))


class TestVoice(TTSProvider):
    __test__ = False
    meta = ProviderMeta(id='tts.test', kind='tts', name='Test')

    def __init__(self, silent_attempts=0):
        super().__init__()
        self.calls = 0
        self.silent_attempts = silent_attempts

    async def synthesize(self, req: TTSRequest, out_wav: Path, progress):
        self.calls += 1
        if self.calls <= self.silent_attempts:
            await ffmpeg.silent_wav(out_wav, .5)
        else:
            await ffmpeg.tone_wav(out_wav, .5)


async def source(project):
    path = store.resolve(project, 'audio/vocals.wav')
    await ffmpeg.tone_wav(path, 4)
    return path


async def test_silent_candidate_retried_before_acceptance(project):
    seg = Segment(start=0, end=1, source_text='こんにちは', translated_text='Hello')
    project.segments = [seg]
    provider = TestVoice(silent_attempts=1)
    take = await stages._synthesize_segment(project, provider, seg, await source(project), lambda *_: None)
    assert provider.calls == 2
    assert seg.active_take_id == take.id
    report = rq.read(project, f'segments/{seg.id}')['synthesis']
    assert report['status'] == 'accepted'
    assert report['attempts'][0]['failures']
    assert report['attempts'][1]['verification']['status'] == 'unavailable'


async def test_all_failed_candidates_never_become_active(project):
    seg = Segment(start=0, end=1, source_text='こんにちは', translated_text='Hello')
    project.segments = [seg]
    with pytest.raises(stages.StageError, match='quality checks failed'):
        await stages._synthesize_segment(project, TestVoice(9), seg, await source(project), lambda *_: None)
    assert not seg.takes
    assert rq.read(project, f'segments/{seg.id}')['synthesis']['status'] == 'failed'


async def test_reference_excludes_longer_overlapping_speech(project):
    speaker = Speaker(name='A', color='#123456')
    project.speakers = [speaker]
    clean = Segment(start=0, end=1.1, speaker_id=speaker.id, source_text='Clean line')
    project.segments = [clean, Segment(start=1.2, end=3.9, speaker_id=speaker.id),
                        Segment(start=2, end=3, speaker_id='other')]
    await stages._build_speaker_reference(project, speaker, await source(project))
    assert speaker.reference_path
    assert store.resolve(project, speaker.reference_path).with_suffix('.txt').read_text() == 'Clean line'


async def test_mix_refuses_missing_line(project):
    project.segments = [Segment(start=0, end=1, source_text='Hello', translated_text='Hi')]
    await ffmpeg.silent_wav(store.resolve(project, 'audio/background.wav'), 4)
    with pytest.raises(stages.StageError, match='unvoiced or stale'):
        await stages.run_mix(project, Job(project_id=project.id, kind='pipeline'), lambda *_: None)


def test_missing_verification_can_be_required(monkeypatch):
    monkeypatch.setattr(rq, 'options', lambda: {'require_verification': True})
    stats = {'status': 'ok', 'duration_s': 1, 'peak_dbfs': -10, 'clipping_fraction': 0}
    assert rq.speech_failures(stats, {'status': 'unavailable', 'reason': 'no model'}, text='Hello')


def test_report_json_is_finite_and_does_not_disclose_credentials(project, monkeypatch):
    project.pipeline.tts.options = {'api_key': 'secret', 'model': 'voice-model'}
    rq.write(project, 'test', {'peak': -float('inf'), **rq.provenance(project)})
    raw = store.resolve(project, 'quality/test.json').read_text()
    assert 'Infinity' not in raw and 'secret' not in raw
    assert rq.read(project, 'test')['peak'] is None


async def test_timing_revision_uses_actual_speech_duration(project, monkeypatch):
    seg = Segment(start=0, end=.25, source_text='Original sentence', translated_text='Long translation')
    project.segments = [seg]
    seen = []

    class DurationVoice(TestVoice):
        async def synthesize(self, req, out_wav, progress):
            self.calls += 1
            await ffmpeg.tone_wav(out_wav, .5 if self.calls == 1 else .2)

    async def shorten(project, segment, text, duration, progress):
        seen.append((text, duration))
        return 'Short'

    monkeypatch.setattr(stages, '_shorten_for_timing', shorten)
    await stages._synthesize_segment(project, DurationVoice(), seg, await source(project), lambda *_: None)
    assert seen[0][0] == 'Long translation' and seen[0][1] == pytest.approx(.5)
    assert seg.translated_text == 'Short'


async def test_resynthesis_clears_old_fitted_verification(project):
    seg = Segment(start=0, end=1, source_text='こんにちは', translated_text='Hello')
    project.segments = [seg]
    rq.segment_report(project, seg, 'fitted', {
        'status': 'accepted', 'verification': {'status': 'ok', 'cer': 0}, 'take_id': 'old'})
    await stages._synthesize_segment(project, TestVoice(), seg, await source(project), lambda *_: None)
    report = rq.read(project, f'segments/{seg.id}')
    assert 'fitted' not in report
    assert report['synthesis']['take_id'] == seg.active_take_id


def test_provenance_redacts_url_credentials(project):
    project.pipeline.tts.options = {'model': 'https://user:password@example.org/model?token=secret'}
    assert rq.provenance(project)['providers']['tts']['settings']['model'] == 'https://example.org/model'


async def test_failed_render_validation_preserves_previous_export(project, monkeypatch):
    final = store.resolve(project, 'render/dubbed.mp4')
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b'previous export')
    store.resolve(project, 'playback.mp4').write_bytes(b'source')
    mix = store.resolve(project, 'audio/dub_mix.wav')
    mix.parent.mkdir(parents=True, exist_ok=True)
    mix.write_bytes(b'new mix')

    async def mux(video, audio, out):
        out.write_bytes(b'invalid candidate')

    async def invalid(*args, **kwargs):
        raise RuntimeError('decode failure')

    monkeypatch.setattr(ffmpeg, 'mux', mux)
    monkeypatch.setattr(ffmpeg, 'validate_render', invalid)
    with pytest.raises(RuntimeError, match='decode failure'):
        await stages.run_render(project, Job(project_id=project.id, kind='pipeline'), lambda *_: None)
    assert final.read_bytes() == b'previous export'
    assert not final.with_name('.dubbed.validating.mp4').exists()
