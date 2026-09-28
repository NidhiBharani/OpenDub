"""Versions: snapshots of dubbing attempts under data/projects/<pid>/versions/<vid>/.

    meta.json        Version (small; listed by GET /versions)
    state.json       VersionState: pipeline, media, speakers, segments, skip ranges
    dub_mix.wav      copy   — the mix stage rewrites the working file in place
    dub_mix.json     copy   (waveform peaks)
    playback_dub.mp4 copy
    dubbed.mp4       copy   (only when render had run)
    takes/<seg>/…    hardlinks (copy on EXDEV) — take files are immutable, but `run_transcribe`
                     rmtree's audio/segments, so the version must hold its own link
    speakers/<id>/reference.wav   hardlinks, same reason

Callers hold ``store.lock(pid)``. Snapshots load a FRESH project from disk (a job's tracked copy
is stale by construction); restore refuses while a job is active because the checkpoint merge in
``store.save_merged`` treats the job's segment list as authoritative.
"""
from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
from pathlib import Path

from .. import store
from ..models import (
    LEGACY_KINDS,
    Project,
    Segment,
    StageState,
    Version,
    VersionKind,
    VersionState,
    now,
)

_OUTPUTS = {
    "dub_mix": ("audio/dub_mix.wav", "dub_mix.wav"),
    "dub_mix_waveform": ("waveforms/dub_mix.json", "dub_mix.json"),
    "playback_dub": ("playback_dub.mp4", "playback_dub.mp4"),
    "dubbed": ("render/dubbed.mp4", "dubbed.mp4"),
}


class VersionError(Exception):
    """User-facing failure (nothing to version, unknown version)."""


def versions_dir(project_id: str) -> Path:
    return store.project_dir(project_id) / "versions"


def _link_or_copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return 0
    try:
        os.link(src, dst)
        return 0  # shares the inode: no extra bytes
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
            raise
        shutil.copy2(src, dst)
        return dst.stat().st_size


def _rel(project_id: str, path: Path) -> str:
    return str(path.relative_to(store.project_dir(project_id).resolve()))


def list_versions(project_id: str) -> list[Version]:
    out: list[Version] = []
    root = versions_dir(project_id)
    if not root.is_dir():
        return out
    for meta in root.glob("*/meta.json"):
        try:
            out.append(Version.model_validate(json.loads(meta.read_text())))
        except (OSError, ValueError):
            continue  # half-written version dir (snapshot interrupted): not listed
    out.sort(key=lambda v: v.created_at, reverse=True)
    return out


def load_version(project_id: str, vid: str) -> tuple[Version, VersionState]:
    d = versions_dir(project_id) / vid
    if not d.resolve().is_relative_to(versions_dir(project_id).resolve()):
        raise VersionError("invalid version id")
    meta = d / "meta.json"
    state = d / "state.json"
    if not meta.exists() or not state.exists():
        raise VersionError(f"version '{vid}' not found")
    return (
        Version.model_validate(json.loads(meta.read_text())),
        VersionState.model_validate(json.loads(state.read_text())),
    )


def _write_json(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def default_label(project: Project, kind: VersionKind) -> str:
    n = len(list_versions(project.id)) + 1
    when = now().strftime("%d %b %H:%M")
    prefix = {"auto": "Run", "manual": "Snapshot", "pre_restore": "Before restore"}[kind]
    return f"{prefix} {n} · {project.target_lang.upper()} · {when}"


def _snapshot_sync(project: Project, *, kind: VersionKind, label: str, job_id: str | None) -> Version:
    pdir = store.project_dir(project.id).resolve()
    version = Version(
        label=label or default_label(project, kind),
        kind=kind,
        source_lang=project.source_lang,
        target_lang=project.target_lang,
        preset=project.pipeline.preset,
        runtime=project.pipeline.runtime,
        providers={k: project.pipeline.choice(k).provider_id for k in LEGACY_KINDS},
        stages={k: v.model_copy(deep=True) for k, v in project.stages.items()},
        segment_count=len(project.segments),
        voiced_count=sum(1 for s in project.segments if s.active_take_id),
        skipped_ranges=len(project.skip_ranges),
        job_id=job_id,
        parent_id=project.active_version_id,
    )
    vdir = versions_dir(project.id) / version.id
    vdir.mkdir(parents=True, exist_ok=True)
    total = 0

    for key, (src_rel, dst_name) in _OUTPUTS.items():
        src = pdir / src_rel
        if src.exists():
            dst = vdir / dst_name
            shutil.copy2(src, dst)
            total += dst.stat().st_size
            version.outputs[key] = _rel(project.id, dst)
    version.has_mix = "dub_mix" in version.outputs
    version.has_render = "dubbed" in version.outputs

    quality_dir = pdir / 'quality'
    if quality_dir.is_dir():
        shutil.copytree(quality_dir, vdir / 'quality')
        total += sum(p.stat().st_size for p in (vdir / 'quality').rglob('*') if p.is_file())

    segments: list[Segment] = []
    for seg in project.segments:
        copy = seg.model_copy(deep=True)
        kept = []
        for take in copy.takes:
            src = pdir / take.path
            if not src.exists():
                continue
            dst = vdir / "takes" / seg.id / src.name
            total += _link_or_copy(src, dst)
            take.path = _rel(project.id, dst)
            kept.append(take)
        copy.takes = kept
        if copy.active_take_id and not any(t.id == copy.active_take_id for t in kept):
            copy.active_take_id = None
        segments.append(copy)

    speakers = []
    for spk in project.speakers:
        copy = spk.model_copy(deep=True)
        if copy.reference_path:
            src = pdir / copy.reference_path
            if src.exists():
                dst = vdir / "speakers" / spk.id / src.name
                total += _link_or_copy(src, dst)
                copy.reference_path = _rel(project.id, dst)
                if src.with_suffix('.txt').exists():
                    shutil.copy2(src.with_suffix('.txt'), dst.with_suffix('.txt'))
            else:
                copy.reference_path = None
        speakers.append(copy)

    state = VersionState(
        pipeline=project.pipeline.model_copy(deep=True),
        media=project.media,
        speakers=speakers,
        segments=segments,
        skip_ranges=[r.model_copy() for r in project.skip_ranges],
    )
    version.bytes = total
    version.summary = summarize(version)
    _write_json(vdir / "state.json", state.model_dump_json(indent=2))
    _write_json(vdir / "meta.json", version.model_dump_json(indent=2))
    return version


def summarize(v: Version) -> str:
    parts = [f"{v.segment_count} lines", f"{v.voiced_count} voiced"]
    if v.skipped_ranges:
        parts.append(f"{v.skipped_ranges} kept original")
    tts = v.providers.get("tts", "")
    if tts:
        parts.append(tts)
    if v.has_render:
        parts.append("rendered")
    return " · ".join(parts)


async def snapshot(
    project_id: str, *, kind: VersionKind, label: str = "", job_id: str | None = None
) -> Version:
    """Snapshot the ON-DISK state (call under store.lock). Requires a mix to exist."""
    project = store.load(project_id)
    if project is None:
        raise VersionError(f"project '{project_id}' not found")
    if not store.resolve(project, "audio/dub_mix.wav").exists():
        raise VersionError("nothing to version yet — run the pipeline through Mix first")
    version = await asyncio.to_thread(_snapshot_sync, project, kind=kind, label=label, job_id=job_id)
    project.active_version_id = version.id
    store.save(project)
    return version


def rename(project_id: str, vid: str, label: str) -> Version:
    version, _ = load_version(project_id, vid)
    version.label = label.strip() or version.label
    _write_json(versions_dir(project_id) / vid / "meta.json", version.model_dump_json(indent=2))
    return version


def delete(project_id: str, vid: str) -> bool:
    d = versions_dir(project_id) / vid
    if not d.resolve().is_relative_to(versions_dir(project_id).resolve()) or not d.is_dir():
        return False
    shutil.rmtree(d)
    return True


def _restore_sync(project: Project, version: Version, state: VersionState) -> Project:
    pdir = store.project_dir(project.id).resolve()
    vdir = versions_dir(project.id) / version.id

    # Reports belong to the snapshotted run, never to the previously active output.
    if (pdir / 'quality').exists():
        shutil.rmtree(pdir / 'quality')
    if (vdir / 'quality').is_dir():
        shutil.copytree(vdir / 'quality', pdir / 'quality')

    project.pipeline = state.pipeline
    project.source_lang = version.source_lang
    project.target_lang = version.target_lang
    project.skip_ranges = state.skip_ranges

    speakers = []
    for spk in state.speakers:
        copy = spk.model_copy(deep=True)
        if copy.reference_path:
            src = pdir / copy.reference_path
            dst = pdir / "speakers" / spk.id / src.name
            if src.exists():
                _link_or_copy(src, dst)
                copy.reference_path = _rel(project.id, dst)
                if src.with_suffix('.txt').exists():
                    shutil.copy2(src.with_suffix('.txt'), dst.with_suffix('.txt'))
            else:
                copy.reference_path = None
        speakers.append(copy)
    project.speakers = speakers

    segments = []
    for seg in state.segments:
        copy = seg.model_copy(deep=True)
        kept = []
        for take in copy.takes:
            src = pdir / take.path
            dst = pdir / "audio" / "segments" / seg.id / src.name
            if src.exists():
                _link_or_copy(src, dst)
                take.path = _rel(project.id, dst)
                kept.append(take)
        copy.takes = kept
        if copy.active_take_id and not any(t.id == copy.active_take_id for t in kept):
            copy.active_take_id = None
        segments.append(copy)
    project.segments = segments

    stages = {k: v.model_copy(deep=True) for k, v in version.stages.items()}
    for key, (dst_rel, _name) in _OUTPUTS.items():
        src_rel = version.outputs.get(key)
        dst = pdir / dst_rel
        if src_rel and (vdir.parent.parent / src_rel).exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vdir.parent.parent / src_rel, dst)
        elif dst.exists():
            dst.unlink()
    (pdir / "render" / "lipsync.mp4").unlink(missing_ok=True)
    if stages.get("lipsync", StageState()).status == "done":
        stages["lipsync"].status = "dirty"
        stages["lipsync"].updated_at = now()
    if not version.has_mix and "mix" in stages:
        stages["mix"].status = "dirty"
    if not version.has_render and stages.get("render", StageState()).status == "done":
        stages["render"].status = "dirty"
    project.stages = stages
    project.active_version_id = version.id
    return project


async def restore(project_id: str, vid: str, *, restore_point: bool = True) -> Project:
    """Make `vid` the working state (call under store.lock, with no active job). The current
    state is snapshotted first as a restore point when it is not already a version."""
    version, state = load_version(project_id, vid)
    project = store.load(project_id)
    if project is None:
        raise VersionError(f"project '{project_id}' not found")
    if (
        restore_point
        and project.active_version_id is None
        and store.resolve(project, "audio/dub_mix.wav").exists()
    ):
        await asyncio.to_thread(
            _snapshot_sync, project, kind="pre_restore",
            label=f"Before restoring {version.label}", job_id=None,
        )
    project = await asyncio.to_thread(_restore_sync, project, version, state)
    store.save(project)
    return project
