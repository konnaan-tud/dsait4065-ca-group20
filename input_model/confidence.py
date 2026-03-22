CONFIDENCE_THRESHOLD_DIFFERENCE = 0.15
CONFIDENCE_THRESHOLD = 0.4
CONFLICT_DIFF_THRESHOLD = 0.25

# Checks if the emotion distribution is confident enough based on top1-top2 difference.
def is_confident(emotion_dict):
    if not emotion_dict:
        return False, None, 0.0, 0.0

    sorted_emotions = sorted(
        emotion_dict.items(), key=lambda x: x[1], reverse=True
    )

    if len(sorted_emotions) < 2:
        return False, sorted_emotions[0][0], sorted_emotions[0][1], 0.0

    top1_label, top1_score = sorted_emotions[0]
    top2_label, top2_score = sorted_emotions[1]

    diff = top1_score - top2_score
    confident = (top1_score >= CONFIDENCE_THRESHOLD) and (diff >= CONFIDENCE_THRESHOLD_DIFFERENCE)

    return confident, top1_label, top1_score, diff

def prune_low_confidence_modalities(modalities):
    if len(modalities) <= 1:
        return modalities

    # Find best confidence
    best_conf = max(m["confidence"] for m in modalities.values())

    pruned = {}

    for name, m in modalities.items():
        diff = best_conf - m["confidence"]

        if diff <= CONFLICT_DIFF_THRESHOLD:
            pruned[name] = m
        else:
            print(f"Discarding {name} modality (confidence too far from best: diff={diff:.2f})")

    return pruned
