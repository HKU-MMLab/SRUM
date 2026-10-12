"""Scoring prompts for the frozen UMM understanding evaluator.

These templates standardize JSON parsing for global/local rewards.
They are not tuned on downstream validation sets (see paper appendix).
"""

GLOBAL_LAYOUT_PROMPT = """\
You are evaluating a generated image against a text prompt.

Prompt: {prompt}

Score the GLOBAL layout and semantic alignment of the whole image
on a continuous scale from 0 (completely misaligned) to 1 (perfectly aligned).
Consider: subject presence, overall composition, and major attributes.

Return ONLY valid JSON:
{{"global_score": <float in [0,1]>, "reason": "<short reason>"}}
"""

LOCAL_REGION_PROMPT = """\
You are evaluating object-level fidelity in a generated image.

Prompt: {prompt}
Region id: {region_id}
Bounding box (xyxy, normalized 0-1): {bbox}

Identify the object that should appear in this region according to the prompt,
then score relevance/quality on a continuous scale from -1 (wrong/harmful)
to 1 (faithful and sharp).

Return ONLY valid JSON:
{{"local_score": <float in [-1,1]>, "object": "<name or none>", "reason": "<short reason>"}}
"""

CANDIDATE_RANK_PROMPT = """\
Given the text prompt and several generated candidates, rank them by
overall faithfulness for text-to-image evaluation (best first).

Prompt: {prompt}
Candidates: {candidate_ids}

Return ONLY valid JSON:
{{"ranking": [<id>, ...], "notes": "<optional>"}}
"""
