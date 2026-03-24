# Applies weighted linear fusion
def fuse_modalities(modalities):
    emotions = list(next(iter(modalities.values()))["probs"].keys())

    confidences = {m: modalities[m]["confidence"] for m in modalities}
    total_c = sum(confidences.values())

    weights = {m: confidences[m] / total_c for m in modalities}

    fused = {e: 0.0 for e in emotions}

    for m in modalities:
        probs = modalities[m]["probs"]
        for e in emotions:
            fused[e] += weights[m] * probs[e]

    final_emotion = max(fused, key=fused.get)

    return final_emotion, fused, weights