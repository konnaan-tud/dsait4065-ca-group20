from transformers import pipeline

def transform_go_emotions_into_ekman(model_output):
    mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
    }

    ekman_scores = {key: 0.0 for key in mapping}
    for item in model_output:
        label, score = item["label"].lower(), item["score"]
        for ekman_label, go_labels in mapping.items():
            if label in go_labels:
                ekman_scores[ekman_label] += score
                break
    total = sum(ekman_scores.values())
    if total > 0:
        ekman_scores = {k: v / total for k, v in ekman_scores.items()}
        
    ekman_scores_refined = sorted([{"label": k, "score": v} for k, v in ekman_scores.items()], key=lambda x: x["score"], reverse=True)

    return ekman_scores_refined

def analyze_text_emotion(text_input):
    print(f"Loading RoBERTa GoEmotions model... (This takes a few seconds on the first run)")
    
    # The pipeline automatically handles the tokenization and model architecture
    classifier = pipeline(task="text-classification", 
                          model="SamLowe/roberta-base-go_emotions", 
                          top_k=None) # top_k=None forces it to return scores for ALL 28 emotions
    
    print("\n" + "="*50)
    print(f"Analyzing Text: '{text_input}'")
    print("="*50)
    
    # Run the model
    results = classifier(text_input)[0]
    results = transform_go_emotions_into_ekman(results)

    # Print the top 3 strongest emotions detected
    print("\nTop 3 Detected Emotions:")
    for i in range(3):
        emotion = results[i]['label']
        score = results[i]['score']
        print(f"  {i+1}. {emotion.capitalize()}: {score:.4f}")

if __name__ == "__main__":
    # Let's test it with the exact emotional sentence from your earlier test!
    test_sentence = "I had a really huge fight with my mom yesterday. We were just sitting in the kitchen, and out of nowhere, she started criticizing my life choices again. I felt so angry and dismissed. I tried to explain how stressed I am, but she just wouldn't listen. It ended with me yelling and storming out of the house. Now, I just feel incredibly guilty and sad about the whole thing."
    
    analyze_text_emotion(test_sentence)