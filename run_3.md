📸 Initializing webcam (Please click 'OK' if Mac asks for permission)...
🧠 Waking up the Multimodal AI Brain... (This will take 10-15 seconds)
  -> Loading Whisper...
  -> Loading DistilRoBERTa Text Emotions (7 Ekman)...
  -> Loading Audeering Prosodic Emotions...
 > tts_models/en/ljspeech/vits is already downloaded.
 > Using model: vits
 > Setting up Audio Processor...
 | > sample_rate:22050
 | > resample:False
 | > num_mels:80
 | > log_func:np.log10
 | > min_level_db:0
 | > frame_shift_ms:None
 | > frame_length_ms:None
 | > ref_level_db:None
 | > fft_size:1024
 | > power:None
 | > preemphasis:0.0
 | > griffin_lim_iters:None
 | > signal_norm:None
 | > symmetric_norm:None
 | > mel_fmin:0
 | > mel_fmax:None
 | > pitch_fmin:None
 | > pitch_fmax:None
 | > spec_gain:20.0
 | > stft_pad_mode:reflect
 | > max_norm:1.0
 | > clip_norm:True
 | > do_trim_silence:False
 | > trim_db:60
 | > do_sound_norm:False
 | > do_amp_to_db_linear:True
 | > do_amp_to_db_mel:True
 | > do_rms_norm:False
 | > db_level:None
 | > stats_path:None
 | > base:10
 | > hop_length:256
 | > win_length:1024

============================================================
✅ SYSTEM READY. Awaiting your turn.
============================================================


------------------------------------------------------------
🟢 TURN 1 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
I0000 00:00:1774385919.365069   32288 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.   
Discarding audio modality (weak neutral: 0.52)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I can't believe my brother just yellled at me for no reason. I feel like I can't do anything right.'

💬 Agent: Sorry to hear that you're going through a tough time with your brother. It sounds like this incident has really shaken your confidence. Can you tell me more about what happened leading up to his outburst?

📖 TEXT MODALITY:
   sadness: 0.30
   surprise: 0.22
   anger: 0.20

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.10
   disgust: 0.19
   fear: 0.04
   joy: 0.02
   sadness: 0.10
   surprise: 0.03
   neutral: 0.52

🎭 VIDEO MODALITY:
   sadness: 0.67
   fear: 0.16
   anger: 0.08

🔎 CONFIDENCE CHECK
Text confident  : False (diff=0.09)
Audio confident : True (diff=0.32)
Face confident  : True (diff=0.50)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - face: sadness (confidence=0.67)
 > Text splitted to sentences.
["Sorry to hear that you're going through a tough time with your brother.", 'It sounds like this incident has really shaken your confidence.', 'Can you tell me more about what happened leading up to his outburst?']
 > Processing time: 5.846611499786377
 > Real-time factor: 0.4511653213025982
💾 Turn stored in Chroma (id: I can't believe my brother just yelled a...)
Stored FINAL emotion in Chroma → sadness ({'anger': 0.07971717444961983, 'disgust': 2.1849839022863085e-05, 'fear': 0.16072515836116222, 'joy': 0.04508536459419354, 'sadness': 0.6651008868532349, 'surprise': 6.858246316889115e-06, 'neutral': 0.049342707656449734})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.26 seconds       
  - RoBERTa (Text Emotion)   : 0.80 seconds       
  - Audeering (Audio Emotion): 0.11 seconds       
  - DeepFace (Video Emotion) : 1.01 seconds (8 frames processed)
  - ChromaDB (Memory Fetch)  : 0.26 seconds       
  - LLM Generation           : 17.53 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 22.96 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 2 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.62)

🧠 Sending profile to LLM...

        [Hidden Context for Agent]

        User message: "He started yelling at me, he does this all the time. I tried talking to him but nothing changes."
        User emotional profile: "Conflicting emotional signals across modalities."

        Emotions detected: anger, sadness

        These signals do not agree.

        Instructions:
        - You noticed this emotional mismatch.    
        - Gently and naturally point out the contrast to the user.
        - Use the emotions provided above (anger, sadness) in your response.
        - Do NOT mention modality names or confidence scores.
        - Frame the exact emotion words conversationally and empathetically, not judgmentally.      
        - End by warmly asking them to clarify how they are truly feeling underneath.
        - Maximum 3 sentences.


============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'He started yelling at me, he does tthis all the time. I tried talking to him but nothing changes.'

💬 Agent: I sense that there's a mix of emotions going on here. It seems like you're feeling both angry at your brother for yelling at you and sad because nothing ever changes despite trying to talk things out. Can you help me understand what's really going on beneath the surface - are you feeling frustrated or defeated?

📖 TEXT MODALITY:
   anger: 0.64
   disgust: 0.14
   neutral: 0.12

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.08
   disgust: 0.13
   fear: 0.03
   joy: 0.02
   sadness: 0.08
   surprise: 0.04
   neutral: 0.62

🎭 VIDEO MODALITY:
   sadness: 0.46
   fear: 0.28
   neutral: 0.25

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.51)
Audio confident : True (diff=0.49)
Face confident  : True (diff=0.18)

🧠 DECISION DEBUG
Decision type: conflict
Number of confident modalities: 2
 - text: anger (confidence=0.64)
 - face: sadness (confidence=0.46)
 > Text splitted to sentences.
["I sense that there's a mix of emotions going on here.", "It seems like you're feeling both angry at your brother for yelling at you and sad because nothing ever changes despite trying to talk things out.", "Can you help me understand what's really going on beneath the surface - are you feeling frustrated or defeated?"]
 > Processing time: 9.436268091201782
 > Real-time factor: 0.4741443454693352
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.47 seconds       
  - RoBERTa (Text Emotion)   : 0.03 seconds       
  - Audeering (Audio Emotion): 0.03 seconds       
  - DeepFace (Video Emotion) : 0.62 seconds (7 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds       
  - LLM Generation           : 12.89 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 17.03 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 3 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
🧠 Resolving previous emotional conflict/no_data from user reply...
{'final_emotion': 'neutral', 'emotion_distribution': {'neutral': 0.9272297024726868, 'disgust': 0.023892389610409737, 'anger': 0.020049339160323143, 'joy': 0.014687541872262955, 'sadness': 0.007293274160474539, 'surprise': 0.0038593194913119078, 'fear': 0.002988508902490139}}
💾 Stored resolved emotion → neutral ({'neutral': 0.9272297024726868, 'disgust': 0.023892389610409737, 'anger': 0.020049339160323143, 'joy': 0.014687541872262955, 'sadness': 0.007293274160474539, 'surprise': 0.0038593194913119078, 'fear': 0.002988508902490139})

🧠 Sending profile to LLM...

🕰️ TEMPORAL MEMORY CHECK (Detour Resolved)        
Past Event Matched: 'I mean, it's not a big deal, I guess. I just go to work, come home. That's kind of how it's been for a while.'
Vector Distance   : 0.039 (MAE)
Result            : ✅ ALIGNMENT (Threshold <= 0.2)
🗣️ User Said: 'It's really not a big deal. I'll juust let it go.'

💬 Agent: Thank you for clarifying that it's not a big deal anymore. Now that we've got that behind us, I'm curious: what do you think your brother might have been upset about before he started yelling at you?
 > Text splitted to sentences.
["Thank you for clarifying that it's not a big deal anymore.", "Now that we've got that behind us, I'm curious: what do you think your brother might have been upset about before he started yelling at you?"]
 > Processing time: 5.5201873779296875
 > Real-time factor: 0.4527739691827967

🔄 [Semantic Memory] Background thread summarizing recent turns...

------------------------------------------------------------
🟢 TURN 4 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding text modality (weak neutral: 0.63)
Discarding audio modality (weak neutral: 0.53)

🧠 Sending profile to LLM...

            [Hidden Context for Agent]

            User message: "I think he was upset because he failed his test."

            No clear emotional signal detected.   

            Instructions:
            - Be honest that you're not fully sure how they're feeling.
            - Ask a gentle, open-ended clarification question.
            - Do NOT guess or force an emotion.   
            - Keep it natural and under 3 sentences.


✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 27.03 seconds)

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I think he was upset because he faiiled his test.'

💬 Agent: I'm not entirely sure how you feel about your brother's test results, but can you tell me more about what that means to you? How do you think his failure might be impacting your relationship with him?

📖 TEXT MODALITY:
   neutral: 0.63
   anger: 0.18
   surprise: 0.09

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.05
   disgust: 0.12
   fear: 0.04
   joy: 0.01
   sadness: 0.21
   surprise: 0.03
   neutral: 0.53

🎭 VIDEO MODALITY:
   sadness: 0.34
   neutral: 0.34
   fear: 0.32

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.45)
Audio confident : True (diff=0.32)
Face confident  : False (diff=0.01)

🧠 DECISION DEBUG
Decision type: no_data
Number of confident modalities: 0
 > Text splitted to sentences.
["I'm not entirely sure how you feel about your brother's test results, but can you tell me more about what that means to you?", 'How do you think his failure might be impacting your relationship with him?']
 > Processing time: 8.231443881988525
 > Real-time factor: 0.6738718426913054
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 4.40 seconds       
  - RoBERTa (Text Emotion)   : 0.02 seconds       
  - Audeering (Audio Emotion): 0.02 seconds       
  - DeepFace (Video Emotion) : 0.43 seconds (3 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds       
  - LLM Generation           : 32.94 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 37.81 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 5 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
🧠 Resolving previous emotional conflict/no_data from user reply...
{'final_emotion': 'neutral', 'emotion_distribution': {'neutral': 0.8992156386375427, 'anger': 0.03730359673500061, 'surprise': 0.02313782274723053, 'sadness': 0.017074666917324066, 'disgust': 0.013957438990473747, 'joy': 0.004970632493495941, 'fear': 0.004340230952948332}}
💾 Stored resolved emotion → neutral ({'neutral': 0.8992156386375427, 'anger': 0.03730359673500061, 'surprise': 0.02313782274723053, 'sadness': 0.017074666917324066, 'disgust': 0.013957438990473747, 'joy': 0.004970632493495941, 'fear': 0.004340230952948332})

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:

💬 Agent: Thank you for clarifying that again. So, going back to what your brother's test results mean to you, can you help me understand how this might be affecting your relationship with him in the long run?
 > Text splitted to sentences.
['Thank you for clarifying that again.', "So, going back to what your brother's test results mean to you, can you help me understand how this might be affecting your relationship with him in the long run?"]
 > Processing time: 7.530977964401245
 > Real-time factor: 0.5832656517472443

------------------------------------------------------------
🟢 TURN 6 | Press [ENTER] to start speaking (or type 'q' to quit): q

👋 Wrapping up the conversation... Please wait a moment.

============================================================
💬 Agent: Thank you so much for sharing all of this with me today. It takes a lot of courage to talk about difficult experiences like your brother yelling at you without warning. I want to acknowledge that you've been through some tough times lately, from feeling frustrated and powerless to eventually letting go of the issue. It's amazing how far you've come in just accepting what happened. Remember, it's okay to not have all the answers or control every situation. You're doing your best, and that's something to be proud of. Take care of yourself, and feel free to reach out whenever you need someone to talk to.
============================================================