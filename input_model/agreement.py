# Defines agreement and detects conflict
def analyze_agreement(modalities):
    if len(modalities) == 0:
        return "no_data", None, {}

    tops = [m["top"] for m in modalities.values()]
    counts = {e: tops.count(e) for e in set(tops)}
    max_count = max(counts.values())
    dominant = max(counts, key=counts.get)

    if max_count == len(modalities):
        # Full agreement — fuse all
        return "full_agreement", dominant, modalities

    elif max_count >= 2:
        # Partial agreement — only keep the agreeing ones
        agreeing = {
            name: m for name, m in modalities.items()
            if m["top"] == dominant
        }
        discarded = [name for name in modalities if name not in agreeing]
        print(f"Partial agreement on '{dominant}', discarding: {discarded}")
        return "partial_agreement", dominant, agreeing

    else:
        return "conflict", None, {}