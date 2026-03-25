============================================================
✅ SYSTEM READY. Awaiting your turn.
============================================================


------------------------------------------------------------
🟢 TURN 1 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
I0000 00:00:1774384905.210232   35780 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Discarding audio modality (confidence too far from best: diff=0.26)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I'm so angry right now my manager kkeeps undermining me and I've had enough'

💬 Agent: "Hey, it sounds like things are really frustrating for you with your manager right now. What's making this situation feel different from any other time when you felt fed up?"

📖 TEXT MODALITY:
   anger: 0.99
   fear: 0.00
   sadness: 0.00

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.07
   fear: 0.02
   joy: 0.02
   sadness: 0.09
   surprise: 0.03
   neutral: 0.73

🎭 VIDEO MODALITY:
   fear: 0.33
   neutral: 0.32
Face confident  : False (diff=0.01)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: anger (confidence=0.99)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'I just had the worst day. My manager humiliated me in front of the whole team and I'm so done with it.'
Past Dominant Emo : disgust
Vector Distance   : 0.274 (MAE)
   ⚠️ Memory vs Present Events Contradiction Detected (Threshold > 0.2)
 > Text splitted to sentences.
['"Hey, it sounds like things are really frustrating for you with your manager right now. What\'s making this situation feel different from any other time when you felt fed up?"']
 > Processing time: 4.990746974945068
 > Real-time factor: 0.46569660605634583
💾 Turn stored in Chroma (id: I'm so angry right now my manager keeps ...)
Stored FINAL emotion in Chroma → anger ({'anger': 0.9906268119812012, 'fear': 0.0030327157583087683, 'sadness': 0.0020122367423027754, 'neutral': 0.0016193721676245332, 'surprise': 0.0011431670282036066, 'disgust': 0.001022728276439011, 'joy': 0.0005

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.17 seconds       
  - RoBERTa (Text Emotion)   : 0.85 seconds       
  - Audeering (Audio Emotion): 0.10 seconds       
  - DeepFace (Video Emotion) : 0.97 seconds (6 frames processed)
  - ChromaDB (Memory Fetch)  : 0.32 seconds       
  - LLM Generation           : 20.53 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 25.94 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 2 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.59)
Discarding face modality (confidence too far from best: diff=0.40)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'He talked over me again in the meetting. I said something and he just ignored me completely. I'm furious.'

💬 Agent: "I can imagine how disappointing that must have been to feel unheard in a meeting. How does this experience compare to others where you've felt dismissed or overlooked?"

📖 TEXT MODALITY:
   anger: 0.98
   disgust: 0.01
   neutral: 0.01

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.05
   disgust: 0.11
   fear: 0.03
   joy: 0.01
   sadness: 0.18
   surprise: 0.03
   neutral: 0.59

🎭 VIDEO MODALITY:
   sadness: 0.58
   fear: 0.25
   neutral: 0.16

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.97)
Audio confident : True (diff=0.41)
Face confident  : True (diff=0.33)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: anger (confidence=0.98)
 > Text splitted to sentences.
['"I can imagine how disappointing that must have been to feel unheard in a meeting. How does this experience compare to others where you\'ve felt dismissed or overlooked?"']
 > Processing time: 4.647720575332642
 > Real-time factor: 0.46330125988284243
💾 Turn stored in Chroma (id: He talked over me again in the meeting. ...)
Stored FINAL emotion in Chroma → anger ({'anger': 0.9804233312606812, 'disgust': 0.0057341367937624454, 'neutral': 0.0053835236467421055, 'sadness': 0.002934937132522464, 'fear': 0.0028747550677508116, 'surprise': 0.0021003626752644777, 'joy': 0.0005488033639267087})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.70 seconds       
  - RoBERTa (Text Emotion)   : 0.02 seconds       
  - Audeering (Audio Emotion): 0.03 seconds       
  - DeepFace (Video Emotion) : 0.64 seconds (8 frames processed)
  - ChromaDB (Memory Fetch)  : 0.20 seconds       
  - LLM Generation           : 7.55 seconds       
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 12.15 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 3 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding face modality (confidence too far from best: diff=0.46)

🧠 Sending profile to LLM...

        [Hidden Context for Agent]

        User message: "I don't care right now. I hate this. I'm so angry. I can't even think straight"
        User emotional profile: "Conflicting emotional signals across modalities."

        Emotions detected: anger, neutral

        These signals do not agree.

        Instructions:
        - You noticed this emotional mismatch.    
        - Gently and naturally point out the contrast to the user.
        - Use the emotions provided above (anger, neutral) in your response.
        - Do NOT mention modality names or confidence scores.
        - Frame the exact emotion words conversationally and empathetically, not judgmentally.      
        - End by warmly asking them to clarify how they are truly feeling underneath.
        - Maximum 3 sentences.


============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I don't care right now. I hate thiss. I'm so angry. I can't even think straight'      

💬 Agent: "I can sense that you're really upset right now, but there's a part of me that wonders if you're also feeling stuck or numb alongside the anger. Can you help me understand what's going on beneath the surface for you?"

📖 TEXT MODALITY:
   anger: 0.91
   disgust: 0.05
   sadness: 0.02

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.06
   fear: 0.02
   joy: 0.02
   sadness: 0.05
   surprise: 0.04
   neutral: 0.79

🎭 VIDEO MODALITY:
   sadness: 0.45
   neutral: 0.28
   fear: 0.26

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.85)
Audio confident : True (diff=0.73)
Face confident  : True (diff=0.16)

🧠 DECISION DEBUG
Decision type: conflict
Number of confident modalities: 2
 - text: anger (confidence=0.91)
 - audio: neutral (confidence=0.79)
 > Text splitted to sentences.
['"I can sense that you\'re really upset right now, but there\'s a part of me that wonders if you\'re also feeling stuck or numb alongside the anger. Can you help me understand what\'s going on beneath the surface for you?"']
 > Processing time: 5.731830358505249
 > Real-time factor: 0.44960889708094065

🔄 [Semantic Memory] Background thread summarizing recent turns...
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.23 seconds       
  - RoBERTa (Text Emotion)   : 0.02 seconds       
  - Audeering (Audio Emotion): 0.03 seconds       
  - DeepFace (Video Emotion) : 0.58 seconds (6 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds       
  - LLM Generation           : 11.38 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 15.24 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 4 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
🧠 Resolving previous emotional conflict/no_data from user reply...
{'final_emotion': 'anger', 'emotion_distribution': {'anger': 0.9161553382873535, 'neutral': 0.04281068965792656, 'disgust': 0.015840020030736923, 'sadness': 0.01108610350638628, 'fear': 0.00765905762091279, 'surprise': 0.00467217992991209, 'joy': 0.001776606310158968}}
💾 Stored resolved emotion → anger ({'anger': 0.9161553382873535, 'neutral': 0.04281068965792656, 'disgust': 0.015840020030736923, 'sadness': 0.01108610350638628, 'fear': 0.00765905762091279, 'surprise': 0.00467217992991209, 'joy': 0.001776606310158968})

🧠 Sending profile to LLM...

✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 29.79 seconds)
🗣️ User Said: 'Whatever, I've been angry about thiis for months, nothing ever changes anyway.'       

💬 Agent: "Thank you for clarifying that anger's been simmering beneath the surface for months. Going back to the meeting where your manager talked over you, how does this experience compare to other times when you felt like your input wasn't valued?"
 > Text splitted to sentences.
['"Thank you for clarifying that anger\'s been simmering beneath the surface for months. Going back to the meeting where your manager talked over you, how does this experience compare to other times when you felt like your input wasn\'t valued?"']   
 > Processing time: 7.647642135620117
 > Real-time factor: 0.5412317988061148

------------------------------------------------------------
🟢 TURN 5 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.64)
Discarding face modality (confidence too far from best: diff=0.34)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user is experiencing frustration and anger towards their manager due to being undermined and ignored in meetings.
• The specific incident that triggered this emotion was when the manager talked over them again, ignoring their input completely.
• The user feels dismissed and overlooked, which is a recurring experience for them.

2. Emotional Trajectory:
• The user's emotional state has shifted from feeling fed up to intense anger, with a sense of helplessness and frustration.
• Beneath the surface, there may be feelings of stuckness or numbness accompanying the anger, suggesting that the user might be struggling with underlying emotions and possibly feeling overwhelmed by their situation.

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I don't know, I'm not even angry annymore, I just feel nothing.'

💬 Agent: "I totally get why you'd feel like you're hitting a wall after months of feeling unheard and ignored. It makes perfect sense that you'd reach a point where all the emotions just feel numb. Can you tell me what it's like to feel this way - is it more like a heavy weight or an empty space?" 

📖 TEXT MODALITY:
   anger: 0.86
   neutral: 0.06
   disgust: 0.05

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.08
   fear: 0.03
   joy: 0.01
   sadness: 0.17
   surprise: 0.03
   neutral: 0.64

🎭 VIDEO MODALITY:
   sadness: 0.52
   neutral: 0.32
   fear: 0.12

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.80)
Audio confident : True (diff=0.46)
Face confident  : True (diff=0.19)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: anger (confidence=0.86)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'Initial Event: I don't care right now. I hate this. I'm so angry. I can't even think straight | Clarification: Whatever, I've been angry about this for months, nothing ever changes anyway.'
Past Dominant Emo : anger
Vector Distance   : 0.018 (MAE)
   ✅ Memory & Present Events Alignment Detected (Threshold <= 0.2)
 > Text splitted to sentences.
['"I totally get why you\'d feel like you\'re hitting a wall after months of feeling unheard and ignored. It makes perfect sense that you\'d reach a point where all the emotions just feel numb. Can you tell me what it\'s like to feel this way - is it more like a heavy weight or an empty space?"']   
 > Processing time: 7.955097913742065
 > Real-time factor: 0.4696131639484165
💾 Turn stored in Chroma (id: I don't know, I'm not even angry anymore...)
Stored FINAL emotion in Chroma → anger ({'anger': 0.8561545014381409, 'neutral': 0.05811038985848427, 'disgust': 0.050226710736751556, 'sadness': 0.024282053112983704, 'fear': 0.006241607014089823, 'surprise': 0.0030755188781768084, 'joy': 0.0019091623835265636})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 2.98 seconds       
  - RoBERTa (Text Emotion)   : 0.02 seconds       
  - Audeering (Audio Emotion): 0.03 seconds       
  - DeepFace (Video Emotion) : 0.54 seconds (6 frames processed)
  - ChromaDB (Memory Fetch)  : 0.20 seconds       
  - LLM Generation           : 29.65 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 33.42 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 6 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding text modality (weak neutral: 0.64)
Discarding audio modality (weak neutral: 0.66)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user is experiencing frustration and anger towards their manager due to being undermined and ignored in meetings.
• The specific incident that triggered this emotion was when the manager talked over them again, ignoring their input completely.
• The user feels dismissed and overlooked, which is a recurring experience for them.

2. Emotional Trajectory:
• The user's emotional state has shifted from feeling fed up to intense anger, with a sense of helplessness and frustration.
• Beneath the surface, there may be feelings of stuckness or numbness accompanying the anger, suggesting that the user might be struggling with underlying emotions and possibly feeling overwhelmed by their situation.

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'It's more like a heavy weight but yyeah it's fine I just go to work and gone home'    

💬 Agent: "It sounds like you're feeling really drained and disconnected from the situation. Can you tell me what goes through your mind when you're commuting or doing something after work - is there anything that helps you unwind or feel more like yourself?"

📖 TEXT MODALITY:
   neutral: 0.64
   disgust: 0.19
   sadness: 0.06

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.07
   fear: 0.03
   joy: 0.02
   sadness: 0.16
   surprise: 0.04
   neutral: 0.66

🎭 VIDEO MODALITY:
   sadness: 0.53
   fear: 0.24
   neutral: 0.22

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.45)
Audio confident : True (diff=0.50)
Face confident  : True (diff=0.28)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - face: sadness (confidence=0.53)
 > Text splitted to sentences.
['"It sounds like you\'re feeling really drained and disconnected from the situation. Can you tell me what goes through your mind when you\'re commuting or doing something after work - is there anything that helps you unwind or feel more like yourself?"']
 > Processing time: 6.144798755645752
 > Real-time factor: 0.43811375576203127

🔄 [Semantic Memory] Background thread summarizing recent turns...
💾 Turn stored in Chroma (id: It's more like a heavy weight but yeah i...)
Stored FINAL emotion in Chroma → sadness ({'anger': 0.010927626459961528, 'disgust': 1.1702813304410254e-05, 'fear': 0.24452494179523176, 'joy': 0.0003691131200507128, 'sadness': 0.5287187990122758, 'surprise': 3.834618461962562e-06, 'neutral': 0.2154439821807137})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.22 seconds       
  - RoBERTa (Text Emotion)   : 0.02 seconds       
  - Audeering (Audio Emotion): 0.03 seconds       
  - DeepFace (Video Emotion) : 0.58 seconds (7 frames processed)
  - ChromaDB (Memory Fetch)  : 0.19 seconds       
  - LLM Generation           : 23.84 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 27.88 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 7 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
Discarding face modality (confidence too far from best: diff=0.21)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user is experiencing frustration and anger towards their manager due to being undermined and ignored in meetings.
• The specific incident that triggered this emotion was when the manager talked over them again, ignoring their input completely.
• The user feels dismissed and overlooked, which is a recurring experience for them.

2. Emotional Trajectory:
• The user's emotional state has shifted from feeling fed up to intense anger, with a sense of helplessness and frustration.
• Beneath the surface, there may be feelings of stuckness or numbness accompanying the anger, suggesting that the user might be struggling with underlying emotions and possibly feeling overwhelmed by their situation.

        [Hidden Context for Agent]

        User message: "Yeah, I usually like to go to the gym after work."
        User emotional profile: "Conflicting emotional signals across modalities."

        Emotions detected: joy, neutral

        These signals do not agree.

        Instructions:
        - You noticed this emotional mismatch.    
        - Gently and naturally point out the contrast to the user.
        - Use the emotions provided above (joy, neutral) in your response.
        - Do NOT mention modality names or confidence scores.
        - Frame the exact emotion words conversationally and empathetically, not judgmentally.      
        - End by warmly asking them to clarify how they are truly feeling underneath.
        - Maximum 3 sentences.


✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 34.01 seconds)

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'Yeah, I usually like to go to the ggym after work.'

💬 Agent: "I can sense that going to the gym after work brings you some joy, but it seems like there's a part of you that's also feeling pretty neutral about it. Can you help me understand what's going on beneath the surface for you - are you really trying to escape or find some release from your daily struggles?"

📖 TEXT MODALITY:
   joy: 0.75
   neutral: 0.14
   disgust: 0.06

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.06
   fear: 0.02
   joy: 0.02
   sadness: 0.06
   surprise: 0.03
   neutral: 0.78

🎭 VIDEO MODALITY:
   sadness: 0.57
   neutral: 0.26
   fear: 0.16

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.61)
Audio confident : True (diff=0.72)
Face confident  : True (diff=0.31)

🧠 DECISION DEBUG
Decision type: conflict
Number of confident modalities: 2
 - text: joy (confidence=0.75)
 - audio: neutral (confidence=0.78)
 > Text splitted to sentences.
['"I can sense that going to the gym after work brings you some joy, but it seems like there\'s a part of you that\'s also feeling pretty neutral about it. Can you help me understand what\'s going on beneath the surface for you - are you really trying to escape or find some release from your daily struggles?"']
 > Processing time: 10.397900104522705
 > Real-time factor: 0.592303810256907
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 2.87 seconds       
  - RoBERTa (Text Emotion)   : 0.02 seconds       
  - Audeering (Audio Emotion): 0.02 seconds       
  - DeepFace (Video Emotion) : 0.54 seconds (4 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds       
  - LLM Generation           : 54.03 seconds      
  -------------------------------------------     
  - TOTAL PIPELINE LATENCY   : 57.49 seconds      
============================================================


------------------------------------------------------------
🟢 TURN 8 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking... 


✅ Recording stopped.

Processing Turn... Please wait.
🧠 Resolving previous emotional conflict/no_data from user reply...
{'final_emotion': 'anger', 'emotion_distribution': {'anger': 0.9301698207855225, 'neutral': 0.027522485703229904, 'sadness': 0.016118301078677177, 'surprise': 0.015435631386935711, 'disgust': 0.00539745669811964, 'fear': 0.0032750656828284264, 'joy': 0.0020812363363802433}}
💾 Stored resolved emotion → anger ({'anger': 0.9301698207855225, 'neutral': 0.027522485703229904, 'sadness': 0.016118301078677177, 'surprise': 0.015435631386935711, 'disgust': 0.00539745669811964, 'fear': 0.0032750656828284264, 'joy': 0.0020812363363802433})

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user's anger towards their manager has been ongoing for months, with feelings of frustration and helplessness.
• The specific incident where the manager talked over them again triggered a sense of being dismissed and overlooked, a recurring experience.
• The user feels stuck, numb, or disconnected from the situation, with no clear emotional response (anger or otherwise).
• The user's coping mechanism appears to be a form of disconnection and avoidance, possibly as a way to deal with the emotional weight.

2. Emotional Trajectory:
till simmering beneath the surface. Going back to what goes through your mind after work, I'm curious - do you find yourself replaying the same thoughts or emotions from your daily struggles, or is there a way you've learned to leave them behind?"
 > Text splitted to sentences.
['"Thank you for clarifying that anger\'s still simmering beneath the surface. Going back to what goes through your mind after work, I\'m curious - do you find yourself replaying the same thoughts or emotions from your daily struggles, or is there a way you\'ve learned to leave them behind?"']
 > Processing time: 10.14122986793518
 > Real-time factor: 0.5913714895167

------------------------------------------------------------
🟢 TURN 9 | Press [ENTER] to start speaking (or type 'q' to quit): q

👋 Wrapping up the conversation... Please wait a moment.

============================================================
💬 Agent: "Thank you so much for sharing all of this with me today. I can imagine how overwhelming it must have felt to deal with months of frustration and helplessness, especially after being dismissed again by your manager. It's no wonder you'd feel stuck and disconnected from the situation, like a heavy weight is holding you back. I want you to know that I've been listening, and I believe you're stronger than this feeling of numbness. Remember, it's okay to not be okay, but acknowledging these emotions is the first step towards healing. Take care of yourself, and we can pick up where we left off whenever you're ready."
============================================================

💾 Saved Final Semantic Summary to final_summary_session_20260324_215223.json