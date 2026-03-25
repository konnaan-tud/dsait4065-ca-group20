============================================================
✅ SYSTEM READY. Awaiting your turn.
============================================================


------------------------------------------------------------
🟢 TURN 1 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
I0000 00:00:1774383393.760425   22264 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Discarding audio modality (weak neutral: 0.67)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I just had the worst day. My manager humiliated me in front of the whole team and I'm so done with it.'

💬 Agent: That's really tough to go through, especially when others are involved. Can you tell me more about what happened before your manager humiliated you?

📖 TEXT MODALITY:
   disgust: 0.92
   anger: 0.03
   sadness: 0.02

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.09
   fear: 0.03
   joy: 0.01
   sadness: 0.12
   surprise: 0.03
   neutral: 0.67

🎭 VIDEO MODALITY:
   neutral: 0.38
   sadness: 0.26
   fear: 0.23

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.89)
Audio confident : True (diff=0.55)
Face confident  : False (diff=0.12)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: disgust (confidence=0.92)
 > Text splitted to sentences.
["That's really tough to go through, especially when others are involved.", 'Can you tell me more about what happened before your manager humiliated you?']
 > Processing time: 7.089113235473633
 > Real-time factor: 0.7233588166472011
💾 Turn stored in Chroma (id: I just had the worst day. My manager hum...)
Stored FINAL emotion in Chroma → disgust ({'disgust': 0.9233213663101196, 'anger': 0.03328053280711174, 'sadness': 0.0229186974465847, 'neutral': 0.013773574493825436, 'fear': 0.003740947460755706, 'surprise': 0.0016746849287301302, 'joy': 0.0012903333408758044})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 4.67 seconds
  - RoBERTa (Text Emotion)   : 1.23 seconds
  - Audeering (Audio Emotion): 0.15 seconds
  - DeepFace (Video Emotion) : 1.80 seconds (8 frames processed)
  - ChromaDB (Memory Fetch)  : 0.54 seconds
  - LLM Generation           : 22.35 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 30.75 seconds
============================================================


------------------------------------------------------------
🟢 TURN 2 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'This isn't the first time either, he always does this. I've told him how I feel  and nothing changes.'

💬 Agent: It sounds like you've already tried to address the issue with your manager before, but unfortunately, it hasn't led to any positive change. What's been going through your mind since this keeps happening?

📖 TEXT MODALITY:
   neutral: 0.47
   sadness: 0.32
   disgust: 0.14

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.06
   fear: 0.02
   joy: 0.02
   sadness: 0.08
   surprise: 0.03
   neutral: 0.75

🎭 VIDEO MODALITY:
   sadness: 0.45
   neutral: 0.40
   fear: 0.13

🔎 CONFIDENCE CHECK
Text confident  : False (diff=0.15)
Audio confident : True (diff=0.67)
Face confident  : False (diff=0.05)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - audio: neutral (confidence=0.75)
 > Text splitted to sentences.
["It sounds like you've already tried to address the issue with your manager before, but unfortunately, it hasn't led to any positive change.", "What's been going through your mind since this keeps happening?"]
 > Processing time: 10.267963171005249
 > Real-time factor: 0.7418756812960894
💾 Turn stored in Chroma (id: This isn't the first time either, he alw...)
Stored FINAL emotion in Chroma → neutral ({'anger': 0.03434514870717466, 'disgust': 0.06278476660477547, 'fear': 0.02052000953272262, 'joy': 0.01819351851965961, 'sadness': 0.08292879038647105, 'surprise': 0.03305466340753596, 'neutral': 0.7481731028416606})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.46 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.04 seconds
  - DeepFace (Video Emotion) : 0.70 seconds (8 frames processed)
  - ChromaDB (Memory Fetch)  : 0.21 seconds
  - LLM Generation           : 10.01 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 14.45 seconds
============================================================


------------------------------------------------------------
🟢 TURN 3 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding face modality (weak neutral: 0.63)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I don't want advice right now, I just want someone to actually listen for once.'

💬 Agent: You're feeling like you need someone to genuinely hear and understand your perspective without trying to fix the problem. Is there something specific that you wish people would do differently when they're listening to you?

📖 TEXT MODALITY:
   neutral: 0.95
   joy: 0.02
   disgust: 0.01

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.06
   fear: 0.02
   joy: 0.03
   sadness: 0.04
   surprise: 0.05
   neutral: 0.76

🎭 VIDEO MODALITY:
   neutral: 0.63
   sadness: 0.19
   anger: 0.10

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.94)
Audio confident : True (diff=0.71)
Face confident  : True (diff=0.44)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 2
 - text: neutral (confidence=0.95)
 - audio: neutral (confidence=0.76)
 > Text splitted to sentences.
["You're feeling like you need someone to genuinely hear and understand your perspective without trying to fix the problem.", "Is there something specific that you wish people would do differently when they're listening to you?"]
 > Processing time: 9.73507571220398
 > Real-time factor: 0.7045557827896812

🔄 [Semantic Memory] Background thread summarizing recent turns...
💾 Turn stored in Chroma (id: I don't want advice right now, I just wa...)
Stored FINAL emotion in Chroma → neutral ({'neutral': 0.8678362941303561, 'joy': 0.0226159142365791, 'disgust': 0.03153325669630833, 'anger': 0.023104901817419095, 'sadness': 0.021598642280040166, 'surprise': 0.0247308028581335, 'fear': 0.00858014155837011})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 2.96 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.02 seconds
  - DeepFace (Video Emotion) : 0.53 seconds (6 frames processed)
  - ChromaDB (Memory Fetch)  : 0.21 seconds
  - LLM Generation           : 9.82 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 13.57 seconds
============================================================


------------------------------------------------------------
🟢 TURN 4 | Press [ENTER] to start speaking (or type 'q' to quit):
✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 36.53 seconds)


🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user had a humiliating experience at work due to their manager's behavior in front of the team.
• This is not the first time the user has experienced this kind of treatment from their manager, despite having previously addressed the issue.
• The user feels frustrated and unheard by others, particularly their manager.
• The user values genuine listening and understanding without being given advice.

2. Emotional Trajectory:
• The user's emotional state is characterized by frustration, anger, and a sense of helplessness due to their ongoing struggles at work.
• As the conversation progresses, the user begins to express a deeper desire for empathy and validation, suggesting that they are open to exploring their emotions and finding support.       

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I mean, it's not a big deal, I guess. I just go to work, come home. That's kind  of how it's been for a while.'

💬 Agent: It sounds like you're trying to downplay the situation and move on from the frustration. What do you usually do to unwind or take your mind off things after a long day at work?    

📖 TEXT MODALITY:
   neutral: 0.84
   disgust: 0.09
   anger: 0.03

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.06
   fear: 0.02
   joy: 0.02
   sadness: 0.12
   surprise: 0.04
   neutral: 0.70

🎭 VIDEO MODALITY:
   neutral: 0.82
   sadness: 0.11
   fear: 0.06

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.74)
Audio confident : True (diff=0.58)
Face confident  : True (diff=0.71)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 3
 - text: neutral (confidence=0.84)
 - audio: neutral (confidence=0.70)
 - face: neutral (confidence=0.82)
 > Text splitted to sentences.
["It sounds like you're trying to downplay the situation and move on from the frustration.", 'What do you usually do to unwind or take your mind off things after a long day at work?']       
 > Processing time: 6.424436569213867
 > Real-time factor: 0.5451100016591467
💾 Turn stored in Chroma (id: I mean, it's not a big deal, I guess. I ...)
Stored FINAL emotion in Chroma → neutral ({'neutral': 0.7917699004446458, 'disgust': 0.05068505485372568, 'anger': 0.021359424481784902, 'joy': 0.012399743377493763, 'sadness': 0.08100980964777785, 'fear': 0.029726093373489006, 'surprise': 0.013049983390146657})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 5.27 seconds
  - RoBERTa (Text Emotion)   : 0.04 seconds
  - Audeering (Audio Emotion): 0.05 seconds
  - DeepFace (Video Emotion) : 1.09 seconds (8 frames processed)
  - ChromaDB (Memory Fetch)  : 0.27 seconds
  - LLM Generation           : 19.46 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 26.17 seconds
============================================================


------------------------------------------------------------
🟢 TURN 5 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.52)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user had a humiliating experience at work due to their manager's behavior in front of the team.
• This is not the first time the user has experienced this kind of treatment from their manager, despite having previously addressed the issue.
• The user feels frustrated and unheard by others, particularly their manager.
• The user values genuine listening and understanding without being given advice.

2. Emotional Trajectory:
• The user's emotional state is characterized by frustration, anger, and a sense of helplessness due to their ongoing struggles at work.
• As the conversation progresses, the user begins to express a deeper desire for empathy and validation, suggesting that they are open to exploring their emotions and finding support.       

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I don't really feel much about it anymore. It's just routine.'

💬 Agent: It seems like you've developed a sense of detachment, and the situation has become more habitual than emotionally draining. What are some things that you used to enjoy or find meaningful before this became your new normal?

📖 TEXT MODALITY:
   neutral: 0.75
   disgust: 0.13
   sadness: 0.07

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.05
   disgust: 0.12
   fear: 0.04
   joy: 0.01
   sadness: 0.23
   surprise: 0.03
   neutral: 0.52

🎭 VIDEO MODALITY:
   neutral: 0.72
   sadness: 0.21
   fear: 0.07

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.62)
Audio confident : True (diff=0.29)
Face confident  : True (diff=0.51)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 2
 - text: neutral (confidence=0.75)
 - face: neutral (confidence=0.72)
 > Text splitted to sentences.
["It seems like you've developed a sense of detachment, and the situation has become more habitual than emotionally draining.", 'What are some things that you used to enjoy or find meaningful before this became your new normal?']
 > Processing time: 6.805705308914185
 > Real-time factor: 0.46629773435653576
💾 Turn stored in Chroma (id: I don't really feel much about it anymor...)
Stored FINAL emotion in Chroma → neutral ({'neutral': 0.7354080279650712, 'disgust': 0.06769088974376812, 'sadness': 0.1368617862614266, 'joy': 0.009282290674151822, 'anger': 0.009720218510679104, 'fear': 0.03844835663796509, 'surprise': 0.0025884504751918763})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 2.97 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.03 seconds
  - DeepFace (Video Emotion) : 0.56 seconds (6 frames processed)
  - ChromaDB (Memory Fetch)  : 0.19 seconds
  - LLM Generation           : 9.37 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 13.15 seconds
============================================================


------------------------------------------------------------
🟢 TURN 6 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.64)
Discarding face modality (weak neutral: 0.54)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user had a humiliating experience at work due to their manager's behavior in front of the team.
• This is not the first time the user has experienced this kind of treatment from their manager, despite having previously addressed the issue.
• The user feels frustrated and unheard by others, particularly their manager.
• The user values genuine listening and understanding without being given advice.

2. Emotional Trajectory:
• The user's emotional state is characterized by frustration, anger, and a sense of helplessness due to their ongoing struggles at work.
• As the conversation progresses, the user begins to express a deeper desire for empathy and validation, suggesting that they are open to exploring their emotions and finding support.       

            [Hidden Context for Agent]

            User message: "I used to get upset about things like this. Now I just don't."      

            No clear emotional signal detected.

            Instructions:
            - Be honest that you're not fully sure how they're feeling.
            - Ask a gentle, open-ended clarification question.
            - Do NOT guess or force an emotion.
            - Keep it natural and under 3 sentences.


============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I used to get upset about things like this. Now I just don't.'

💬 Agent: I get the sense that things have changed for you, but I'm not entirely sure what's going on beneath the surface. Can you tell me more about what you mean by "I used to get upset" – was there a specific incident or moment when you started feeling differently?

📖 TEXT MODALITY:
   neutral: 0.42
   anger: 0.41
   disgust: 0.12

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.05
   disgust: 0.11
   fear: 0.03
   joy: 0.01
   sadness: 0.13
   surprise: 0.03
   neutral: 0.64

🎭 VIDEO MODALITY:
   neutral: 0.54
   fear: 0.33
   sadness: 0.12

🔎 CONFIDENCE CHECK
Text confident  : False (diff=0.02)
Audio confident : True (diff=0.51)
Face confident  : True (diff=0.21)

🧠 DECISION DEBUG
Decision type: no_data
Number of confident modalities: 0
 > Text splitted to sentences.
["I get the sense that things have changed for you, but I'm not entirely sure what's going on beneath the surface.", 'Can you tell me more about what you mean by "I used to get upset" – was there a specific incident or moment when you started feeling differently?']
 > Processing time: 9.490414381027222
 > Real-time factor: 0.5813395554650698

🔄 [Semantic Memory] Background thread summarizing recent turns...
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 2.90 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.03 seconds
  - DeepFace (Video Emotion) : 0.49 seconds (5 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds
  - LLM Generation           : 20.48 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 23.92 seconds
============================================================


------------------------------------------------------------
🟢 TURN 7 | Press [ENTER] to start speaking (or type 'q' to quit):

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
🧠 Resolving previous emotional conflict/no_data from user reply...
{'final_emotion': 'neutral', 'emotion_distribution': {'neutral': 0.3559309244155884, 'sadness': 0.31035706400871277, 'disgust': 0.2414509356021881, 'fear': 0.04347263276576996, 'anger': 0.030913032591342926, 'surprise': 0.014869003556668758, 'joy': 0.003006373532116413}}
💾 Stored resolved emotion → neutral ({'neutral': 0.3559309244155884, 'sadness': 0.31035706400871277, 'disgust': 0.2414509356021881, 'fear': 0.04347263276576996, 'anger': 0.030913032591342926, 'surprise': 0.014869003556668758, 'joy': 0.003006373532116413})

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
• The user had a humiliating experience at work due to their manager's behavior in front of the team.
• This is not the first time the user has experienced this kind of treatment from their manager, despite having previously addressed the issue.
• The user feels frustrated and unheard by others, particularly their manager.
• The user values genuine listening and understanding without being given advice.

2. Emotional Trajectory:
• The user's emotional state is characterized by frustration, anger, and a sense of helplessness due to their ongoing struggles at work.
• As the conversation progresses, the user begins to express a deeper desire for empathy and validation, suggesting that they are open to exploring their emotions and finding support.       

✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 35.36 seconds)
🗣️ User Said: 'Maybe I'm overreacting. I don't know how I feel.'

💬 Agent: Thank you for clarifying your emotions. Now, I'd love to get back to exploring what changed for you – can you tell me if there was a particular incident or moment when things shifted for you?
 > Text splitted to sentences.
['Thank you for clarifying your emotions.', "Now, I'd love to get back to exploring what changed for you – can you tell me if there was a particular incident or moment when things shifted for you?"]
 > Processing time: 6.950460195541382
 > Real-time factor: 0.572266875193002
