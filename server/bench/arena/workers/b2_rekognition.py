"""B2 worker: Amazon Rekognition Video segment detection (SHOT + TECHNICAL_CUE).

The API reads video only from S3: each item is uploaded to ``bucket`` (param or
``OPENDUB_AWS_BUCKET``) under ``prefix``, analysed with ``StartSegmentDetection`` and polled with
``GetSegmentDetection`` (no SNS topic needed), then deleted. Credentials: the standard boto3 chain
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_DEFAULT_REGION``).

Pricing (aws.amazon.com/rekognition/pricing, us-east-1, checked 2026-09-28): shot detection
$0.05/min, technical cues $0.05/min, per started minute of video. Technical cues (black frames,
credits, colour bars, slates) are returned in the payload for the audit viewer; only SHOT segments
become transitions.

params: bucket, prefix (opendub-arena/), region, segment_types ([SHOT, TECHNICAL_CUE]),
min_confidence (50), price_per_min_per_type (0.05), poll_s (5), timeout_s (1800).
"""
from __future__ import annotations

import math
import os
import time
import uuid
from pathlib import Path

from _sdk import serve
from b_common import probe


def load(params: dict, lang: str) -> dict:
    import boto3

    bucket = params.get("bucket") or os.environ.get("OPENDUB_AWS_BUCKET")
    if not bucket:
        raise RuntimeError("set params.bucket or OPENDUB_AWS_BUCKET (Rekognition reads from S3)")
    kw = {"region_name": params["region"]} if params.get("region") else {}
    return {"s3": boto3.client("s3", **kw), "rek": boto3.client("rekognition", **kw),
            "bucket": bucket, "params": params}


def run(state: dict, item: dict, out: Path) -> dict:
    p = state["params"]
    video = item["inputs"]["video"]
    info = probe(video)
    key = f"{p.get('prefix', 'opendub-arena/')}{uuid.uuid4().hex}{Path(video).suffix}"
    state["s3"].upload_file(video, state["bucket"], key)
    types = list(p.get("segment_types", ["SHOT", "TECHNICAL_CUE"]))
    conf = float(p.get("min_confidence", 50))
    try:
        filters = {}
        if "SHOT" in types:
            filters["ShotFilter"] = {"MinSegmentConfidence": conf}
        if "TECHNICAL_CUE" in types:
            filters["TechnicalCueFilter"] = {"MinSegmentConfidence": conf}
        job = state["rek"].start_segment_detection(
            Video={"S3Object": {"Bucket": state["bucket"], "Name": key}}, SegmentTypes=types,
            Filters=filters)["JobId"]
        deadline = time.time() + float(p.get("timeout_s", 1800))
        segments, token = [], None
        while True:
            kw = {"JobId": job, "MaxResults": 1000}
            if token:
                kw["NextToken"] = token
            resp = state["rek"].get_segment_detection(**kw)
            status = resp["JobStatus"]
            if status == "IN_PROGRESS":
                if time.time() > deadline:
                    raise TimeoutError("Rekognition segment detection timed out")
                time.sleep(float(p.get("poll_s", 5)))
                continue
            if status != "SUCCEEDED":
                raise RuntimeError(f"Rekognition job {status}: {resp.get('StatusMessage')}")
            segments += resp.get("Segments", [])
            token = resp.get("NextToken")
            if not token:
                break
    finally:
        state["s3"].delete_object(Bucket=state["bucket"], Key=key)
    shots = sorted((s["StartTimestampMillis"] / 1000, s["EndTimestampMillis"] / 1000)
                   for s in segments if s.get("Type") == "SHOT")
    cues = [{"type": s["TechnicalCueSegment"]["Type"], "start": s["StartTimestampMillis"] / 1000,
             "end": s["EndTimestampMillis"] / 1000,
             "confidence": s["TechnicalCueSegment"].get("Confidence")}
            for s in segments if s.get("Type") == "TECHNICAL_CUE"]
    minutes = math.ceil(max(info["duration_s"], 1e-3) / 60.0)
    return {"shots": [{"start": a, "end": b} for a, b in shots],
            "transitions": [{"start": a, "end": a, "type": "cut"} for a, _b in shots[1:]],
            "technical_cues": cues, "fps": info["fps"], "duration_s": info["duration_s"],
            "_cost_usd": minutes * len(types) * float(p.get("price_per_min_per_type", 0.05))}


if __name__ == "__main__":
    serve(load, run)
