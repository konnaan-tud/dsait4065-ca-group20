from test_audeering import Wav2Small
import librosa, torch
import sys

class EmotionPredictor:

    def __init__(self, device):
        self.device = device
        self.model = Wav2Small.from_pretrained(
            'audeering/wav2small').to(device).eval()

    def predict(self, audio_path):
        signal = torch.from_numpy(librosa.load(audio_path, sr=16000)[0])[None, :]
        with torch.no_grad():
            logits = self.model(signal.to(self.device))
        arousal = logits[:, 0]
        dominance = logits[:, 1]
        valence = logits[:, 2]
        return arousal, dominance, valence
    
if __name__ == "__main__":
    predictor = EmotionPredictor(device="cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) < 2:
        print("Usage: python audeering_example.py <audio_file.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    arousal, dominance, valence = predictor.predict(audio_path)
    print(f"Arousal: {arousal.item():.4f}, Dominance: {dominance.item():.4f}, Valence: {valence.item():.4f}")