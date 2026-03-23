import subprocess
from TTS.api import TTS

text = [
    "I had a really huge fight with my mom yesterday. ",
    "We were just sitting in the kitchen, and out of nowhere, ",
    "she started criticizing my life choices again. ",
    "I felt so angry and dismissed. ",
    "I tried to explain how stressed I am, but she just wouldn't, listen. ",
    "It ended with me yelling and storming out of the house. ",
    "Now, I just feel incredibly guilty and sad about the whole thing.",
]

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# A loop is used to check how long does it take to generate audio for each sentence 
# not counting the time which is needed to load the model
for sentence in text:
    tts.tts_to_file(text=sentence, file_path="output.wav")
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "output.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
