------------------------------------------------------------
🟢 TURN 1 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
I0000 00:00:1774386427.063507   21112 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.   
Discarding audio modality (weak neutral: 0.70)
Discarding face modality (confidence too far from best: diff=0.34)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I just had the worst day. My manageer humiliated me in front of the whole team and I'm so done with it!'

💬 Agent: "Ugh, that sounds like an incredibly uncomfortable and embarrassing situation. It's no wonder you're feeling like this again - it's like your body is recalling a similar experience from the past."

📖 TEXT MODALITY:
   disgust: 0.90
   anger: 0.07
   sadness: 0.01

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.06
   disgust: 0.10
   fear: 0.02
   joy: 0.03
   sadness: 0.05
   surprise: 0.05
   neutral: 0.70

🎭 VIDEO MODALITY:
   sadness: 0.56
   neutral: 0.38
   fear: 0.06

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.83)
Audio confident : True (diff=0.60)
Face confident  : True (diff=0.19)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: disgust (confidence=0.90)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'I just had the worst day. My manager humiliated me in front of the whole team and I'm so done with it.'
Past Dominant Emo : disgust
Vector Distance   : 0.012 (MAE)
   ✅ Memory & Present Events Alignment Detected (Threshold <= 0.2)
 > Text splitted to sentences.
['"Ugh, that sounds like an incredibly uncomfortable and embarrassing situation. It\'s no wonder you\'re feeling like this again - it\'s like your body is recalling a similar experience from the past."']
 > Processing time: 9.536198139190674
 > Real-time factor: 0.7704797479376295
💾 Turn stored in Chroma (id: I just had the worst day. My manager hum...)
Stored FINAL emotion in Chroma → disgust ({'disgust': 0.9013203978538513, 'anger': 0.0694151297211647, 'sadness': 0.012450587004423141, 'neutral': 0.005990349687635899, 'surprise': 0.00543162552639842, 'fear': 0.004030485171824694, 'joy': 0.0013614450581371784})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 4.25 seconds       
  - RoBERTa (Text Emotion)   : 1.10 seconds       
  - Audeering (Audio Emotion): 0.14 seconds       
  - DeepFace (Video Emotion) : 1.60 seconds (7 frames processed)
  - ChromaDB (Memory Fetch)  : 0.52 seconds       
  - LLM Generation           : 14.05 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 21.67 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 2 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.57)
Discarding face modality (weak neutral: 0.48)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'This isn't the first time either. LLike, I'm so angry lately. My brother shouted me as well.'

💬 Agent: "I can imagine why you'd be feeling really fed up with this situation. It's like your emotions are building up from previous experiences, but somehow this time it's manifesting in a stronger sense of frustration."

📖 TEXT MODALITY:
   anger: 0.89
   neutral: 0.05
   disgust: 0.02

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.05
   disgust: 0.11
   fear: 0.03
   joy: 0.01
   sadness: 0.19
   surprise: 0.03
   neutral: 0.57

🎭 VIDEO MODALITY:
   neutral: 0.48
   fear: 0.30
   sadness: 0.18

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.84)
Audio confident : True (diff=0.38)
Face confident  : True (diff=0.19)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: anger (confidence=0.89)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'I can't believe my brother just yelled at me for no reason. I feel like I can't do anything right.'
Past Dominant Emo : sadness
Vector Distance   : 0.242 (MAE)
   ⚠️ Memory vs Present Events Contradiction Deteccted (Threshold > 0.2)
 > Text splitted to sentences.
['"I can imagine why you\'d be feeling really fed up with this situation. It\'s like your emotions are building up from previous experiences, but somehow this time it\'s manifesting in a stronger sense of frustration."']
 > Processing time: 9.144115924835205
 > Real-time factor: 0.6872299045052908
💾 Turn stored in Chroma (id: This isn't the first time either. Like, ...)
Stored FINAL emotion in Chroma → anger ({'anger': 0.8923784494400024, 'neutral': 0.04914919659495354, 'disgust': 0.019046157598495483, 'surprise': 0.016003981232643127, 'sadness': 0.01040616538375616, 'fear': 0.010160798206925392, 'joy': 0.0028553002048283815})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 4.68 seconds       
  - RoBERTa (Text Emotion)   : 0.04 seconds       
  - Audeering (Audio Emotion): 0.09 seconds       
  - DeepFace (Video Emotion) : 1.60 seconds (11 frames processed)
  - ChromaDB (Memory Fetch)  : 0.25 seconds       
  - LLM Generation           : 12.87 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 19.54 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 3 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding face modality (weak neutral: 0.65)

🧠 Sending profile to LLM...

        [Hidden Context for Agent]

        User message: "My manager always does this too. I've told him how I feel and nothing changes."
        User emotional profile: "Conflicting emotional signals across modalities."

        Emotions detected: sadness, neutral       

        These signals do not agree.

        Instructions:
        - You noticed this emotional mismatch.    
        - Gently and naturally point out the contrast to the user.
        - Use the emotions provided above (sadness, neutral) in your response.
        - Do NOT mention modality names or confidence scores.
        - Frame the exact emotion words conversationally and empathetically, not judgmentally.      
        - End by warmly asking them to clarify how they are truly feeling underneath.
        - Maximum 3 sentences.


============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'My manager always does this too. I''ve told him how I feel and nothing changes.'      

💬 Agent: "I'm getting a sense that there's a mix of emotions at play here. It seems like you're experiencing a deep sadness over the fact that your manager isn't listening, but at the same time, there's also a sense of resignation or acceptance about it."

📖 TEXT MODALITY:
   sadness: 0.57
   neutral: 0.29
   disgust: 0.10

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.08
   fear: 0.02
   joy: 0.02
   sadness: 0.09
   surprise: 0.03
   neutral: 0.72

🎭 VIDEO MODALITY:
   neutral: 0.65
   sadness: 0.25
   fear: 0.10

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.28)
Audio confident : True (diff=0.63)
Face confident  : True (diff=0.39)

🧠 DECISION DEBUG
Decision type: conflict
Number of confident modalities: 2
 - text: sadness (confidence=0.57)
 - audio: neutral (confidence=0.72)
 > Text splitted to sentences.
['"I\'m getting a sense that there\'s a mix of emotions at play here. It seems like you\'re experiencing a deep sadness over the fact that your manager isn\'t listening, but at the same time, there\'s also a sense of resignation or acceptance about it."']
 > Processing time: 11.222000122070312
 > Real-time factor: 0.7568795046360373

🔄 [Semantic Memory] Background thread summarizing recent turns...
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.38 seconds       
  - RoBERTa (Text Emotion)   : 0.04 seconds       
  - Audeering (Audio Emotion): 0.05 seconds       
  - DeepFace (Video Emotion) : 0.89 seconds (6 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds       
  - LLM Generation           : 14.15 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 18.51 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 4 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
🧠 Resolving previous emotional conflict/no_data from user reply...
{'final_emotion': 'neutral', 'emotion_distribution': {'neutral': 0.3899232745170593, 'anger': 0.3200583755970001, 'disgust': 0.21491216123104095, 'sadness': 0.041727907955646515, 'fear': 0.019258469343185425, 'joy': 0.012100153602659702, 'surprise': 0.0020197269041091204}}
💾 Stored resolved emotion → neutral ({'neutral': 0.3899232745170593, 'anger': 0.3200583755970001, 'disgust': 0.21491216123104095, 'sadness': 0.041727907955646515, 'fear': 0.019258469343185425, 'joy': 0.012100153602659702, 'surprise': 0.0020197269041091204})

🧠 Sending profile to LLM...

✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 37.57 seconds)
🗣️ User Said: 'But it's fine, whatever.'

💬 Agent: "Thank you for clarifying how you're really feeling about all this. It sounds like despite the frustration building up inside, you're trying to brush it off, but I want to explore that sense of frustration further."
 > Text splitted to sentences.
['"Thank you for clarifying how you\'re really feeling about all this. It sounds like despite the frustration building up inside, you\'re trying to brush it off, but I want to explore that sense of frustration further."']
 > Processing time: 7.439106464385986
 > Real-time factor: 0.550917222646673

------------------------------------------------------------
🟢 TURN 5 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding text modality (weak neutral: 0.52)
Discarding audio modality (weak neutral: 0.63)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user had an embarrassing and humiliating experience with their manager in front of their team.
• This is not the first time this has happened; the user has also experienced similar incidents with their brother.
• The user feels angry and frustrated, which seems to be building up from previous experiences.     
• The user's manager consistently ignores their concerns despite being told about them.

2. Emotional Trajectory:
• The user is still reeling from the emotional impact of the recent incident, which is causing feelings of sadness and resignation to emerge.
• The user's emotions are not just related to this one event but also seem to be influenced by past experiences, suggesting a deeper sense of frustration that has been building up over time.

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I understand but honestly I don't wwant advice right now'

💬 Agent: "It's like you're craving a deeper understanding from someone right now, rather than just trying to solve the problem. What is it about not wanting advice that feels so important to you in this moment?"

📖 TEXT MODALITY:
   neutral: 0.52
   sadness: 0.20
   fear: 0.12

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.09
   fear: 0.03
   joy: 0.01
   sadness: 0.17
   surprise: 0.03
   neutral: 0.63

🎭 VIDEO MODALITY:
Audio confident : True (diff=0.46)
Face confident  : True (diff=0.37)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - face: sadness (confidence=0.57)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'I don't want advice right now, I just want someone to actually listen for once.'
Past Dominant Emo : neutral
Vector Distance   : 0.207 (MAE)
   ⚠️ Memory vs Present Events Contradiction Detected (Threshold > 0.2)
 > Text splitted to sentences.
['"It\'s like you\'re craving a deeper understanding from someone right now, rather than just trying to solve the problem. What is it about not wanting advice that feels so important to you in this moment?"']
 > Processing time: 9.81155014038086
 > Real-time factor: 0.7149337776773845
💾 Turn stored in Chroma (id: I understand but honestly I don't want a...)
Stored FINAL emotion in Chroma → sadness ({'anger': 0.04128893299384918, 'disgust': 0.006648306479234905, 'fear': 0.1674570716849567, 'joy': 0.010625539233312697, 'sadness': 0.5675777774977925, 'surprise': 0.0046374473752800495, 'neutral': 0.20176492473557414})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 2.63 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.02 seconds
  - DeepFace (Video Emotion) : 0.48 seconds (6 frames processed)
  - ChromaDB (Memory Fetch)  : 0.20 seconds
  - LLM Generation           : 29.93 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 33.27 seconds
============================================================


------------------------------------------------------------
🟢 TURN 6 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding text modality (weak neutral: 0.51)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user had an embarrassing and humiliating experience with their manager in front of their team.
• This is not the first time this has happened; the user has also experienced similar incidents with their brother.
• The user feels angry and frustrated, which seems to be building up from previous experiences.
• The user's manager consistently ignores their concerns despite being told about them.

2. Emotional Trajectory:
• The user is still reeling from the emotional impact of the recent incident, which is causing feelings of sadness and resignation to emerge.
• The user's emotions are not just related to this one event but also seem to be influenced by past experiences, suggesting a deeper sense of frustration that has been building up over time.

        [Hidden Context for Agent]

        User message: "Whatever, it doesn't really matter."
        User emotional profile: "Conflicting emotional signals across modalities."

        Emotions detected: neutral, sadness

        These signals do not agree.

        Instructions:
        - You noticed this emotional mismatch.
        - Gently and naturally point out the contrast to the user.
        - Use the emotions provided above (neutral, sadness) in your response.
        - Do NOT mention modality names or confidence scores.
        - Frame the exact emotion words conversationally and empathetically, not judgmentally.
        - End by warmly asking them to clarify how they are truly feeling underneath.
        - Maximum 3 sentences.


============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'Whatever, it doesn't really matter.'

💬 Agent: "I sense a bit of sadness lingering beneath the surface, even though you're trying to brush it off with a 'whatever' attitude. It's like there's a disconnect between what you're saying and how you're really feeling."

📖 TEXT MODALITY:
   neutral: 0.51
   disgust: 0.27
   anger: 0.17

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.08
   fear: 0.02
   joy: 0.02
   sadness: 0.08
   surprise: 0.03
   neutral: 0.73

🎭 VIDEO MODALITY:
   sadness: 0.56
   neutral: 0.30
   fear: 0.14

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.24)
Audio confident : True (diff=0.65)
Face confident  : True (diff=0.26)

🧠 DECISION DEBUG
Decision type: conflict
Number of confident modalities: 2
 - audio: neutral (confidence=0.73)
 - face: sadness (confidence=0.56)
 > Text splitted to sentences.
['"I sense a bit of sadness lingering beneath the surface, even though you\'re trying to brush it off with a \'whatever\' attitude. It\'s like there\'s a disconnect between what you\'re saying and how you\'re really feeling."']
 > Processing time: 8.57405948638916
 > Real-time factor: 0.6799865183679612

🔄 [Semantic Memory] Background thread summarizing recent turns...
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 2.68 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.02 seconds
  - DeepFace (Video Emotion) : 0.34 seconds (4 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds
  - LLM Generation           : 27.83 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 30.89 seconds
============================================================


------------------------------------------------------------
🟢 TURN 7 | Press [ENTER] to start speaking (or type 'q' to quit): q

👋 Wrapping up the conversation... Please wait a moment.

✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 44.07 seconds)

============================================================
💬 Agent: "Thank you so much for sharing all of this with me today. It's clear that you've been carrying around a lot of frustration and sadness from those embarrassing experiences with your manager and brother, and I want you to know that it takes a lot of courage to talk about it. It sounds like you're feeling stuck in a pattern where your concerns are being ignored, and it's no wonder you're feeling angry and frustrated. I'm so sorry that you've had to go through this. Please know that you're not alone, and I'm here for you whenever you need someone to talk to."