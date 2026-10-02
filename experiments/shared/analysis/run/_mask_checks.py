"""Conservative provenance checks for observations omitted from a response."""

from collections import defaultdict


def unretained_observation_requests(trajectories, generations):
    """Identify single-generation responses containing no appended tool tokens.

    A tool may execute before the loop discovers its reply will exceed the
    response budget. An all-one mask is then valid: the recorded response is
    exactly the generated assistant tokens. Require independent generation
    counts and either the complete mask or a complete single GEN span (full
    token dumps may be capped). Never infer omission from cap proximity or a
    termination label. Missing/ambiguous evidence does not exempt a trace.
    """
    by_request = defaultdict(list)
    for generation in generations:
        by_request[generation.get("request_id")].append(generation)
    result = set()
    for trajectory in trajectories:
        request_id = trajectory.get("request_id")
        records = by_request.get(request_id, [])
        if not request_id or len(records) != 1:
            continue
        generation = records[0]
        if trajectory.get("assistant_turns") != 1 or generation.get("assistant_turn") != 1:
            continue
        if trajectory.get("num_masked_tokens") != 0:
            continue
        if trajectory.get("user_turns") != 0:
            continue
        length = generation.get("token_count")
        if type(length) is not int or length <= 0:
            continue
        mask = trajectory.get("response_mask")
        if isinstance(mask, str):
            mask = mask.split(",")
        if mask is None:
            spans = trajectory.get("turn_spans")
            if not isinstance(spans, list) or len(spans) != 1:
                continue
            span = spans[0]
            if not isinstance(span, dict) or not (
                span.get("kind") == "GEN" and span.get("start") == 0 and span.get("end") == length
            ):
                continue
        elif not isinstance(mask, list) or len(mask) != length or any(value not in (1, "1") for value in mask):
            continue
        if not all(value == length for value in (
            trajectory.get("response_len"),
            trajectory.get("num_response_tokens"),
        )):
            continue
        result.add(request_id)
    return result
