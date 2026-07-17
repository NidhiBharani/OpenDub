"""OpenDub benchmarking & quality-assessment harness.

Drives the real dubbing pipeline over registered cases and computes per-stage quality metrics
(reference-free proxies always; ground-truth metrics like WER/chrF/SI-SDR when reference files
exist). Not imported by the `app` package — a standalone tool run via `python -m bench`.
"""
