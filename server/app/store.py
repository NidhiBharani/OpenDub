"""Project persistence. One directory per project under data/projects/<id>/ containing all media
plus manifest.json (the serialized Project). Only this module deals in absolute paths; every path
stored on models is relative to the project dir.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from . import config
from .models import Project, ProjectSummary

_locks: dict[str, asyncio.Lock] = {}


def lock(project_id: str) -> asyncio.Lock:
    return _locks.setdefault(project_id, asyncio.Lock())


def project_dir(project_id: str) -> Path:
    d = config.PROJECTS_DIR / project_id
    if not d.resolve().is_relative_to(config.PROJECTS_DIR.resolve()):
        raise ValueError("invalid project id")
    return d


def resolve(project: Project | str, rel_path: str) -> Path:
    """Absolute path for a project-relative media path (validated against traversal)."""
    pid = project if isinstance(project, str) else project.id
    base = project_dir(pid).resolve()
    p = (base / rel_path).resolve()
    if not p.is_relative_to(base):
        raise ValueError("invalid media path")
    return p


def save(project: Project) -> None:
    d = project_dir(project.id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    tmp.write_text(project.model_dump_json(indent=2))
    tmp.replace(d / "manifest.json")


def load(project_id: str) -> Project | None:
    manifest = project_dir(project_id) / "manifest.json"
    if not manifest.exists():
        return None
    return Project.model_validate(json.loads(manifest.read_text()))


def delete(project_id: str) -> bool:
    d = project_dir(project_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    _locks.pop(project_id, None)
    return True


def list_summaries() -> list[ProjectSummary]:
    out: list[ProjectSummary] = []
    if not config.PROJECTS_DIR.exists():
        return out
    for d in sorted(config.PROJECTS_DIR.iterdir()):
        if (d / "manifest.json").exists():
            try:
                p = Project.model_validate(json.loads((d / "manifest.json").read_text()))
            except Exception:
                continue
            out.append(
                ProjectSummary(
                    id=p.id,
                    name=p.name,
                    created_at=p.created_at,
                    duration=p.media.duration if p.media else 0.0,
                    target_lang=p.target_lang,
                    segment_count=len(p.segments),
                    stages=p.stages,
                )
            )
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out
