# Defines agreement and detects conflict
def analyze_agreement(modalities):
    if len(modalities) == 0:
        return "no_data", None

    tops = [m["top"] for m in modalities.values()]
    counts = {e: tops.count(e) for e in set(tops)}

    max_count = max(counts.values())
    dominant = max(counts, key=counts.get)

    if max_count == len(modalities):
        return "full_agreement", dominant
    elif max_count == 2:
        return "partial_agreement", dominant
    else:
        return "conflict", None