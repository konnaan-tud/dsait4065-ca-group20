CONFIDENCE_THRESHOLD = 0.15

# Checks if the emotion distribution is confident enough based on top1-top2 difference.
def is_confident(emotion_dict, threshold=CONFIDENCE_THRESHOLD):
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
    confident = diff >= threshold

    return confident, top1_label, top1_score, diff