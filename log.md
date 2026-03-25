### Eventhough i said that i am happy in the end the llm thought that i feel uneasy abouth the situation at the moment

```
🗣️ User Said: 'During the jump I was terrified, really. I don't think I've ever been so scared in my life. However, afterwards it was amazing. I must say that it was an unforgettable experience. And I'm so grateful for my friends that they made me do it. I would never do it just by myself without having these motivations that they gave me. So, yeah, summarizing, I don't regret it all. It was great.'

💬 Agent: "I sense that you're reflecting on this experience in a different way now compared to when it first happened. What's changed for you since then? Is there something specific that's making you feel more uneasy or anxious about the experience now?"

📖 TEXT MODALITY:
   fear: 0.97
   neutral: 0.01
   surprise: 0.01

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.07
   fear: 0.03
   joy: 0.01
   sadness: 0.36
   surprise: 0.03
   neutral: 0.46

🎭 VIDEO MODALITY:
   neutral: 0.48
   joy: 0.13
   surprise: 0.13

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.96)
Audio confident : False (diff=0.11)
Face confident  : True (diff=0.34)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: fear (confidence=0.97)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'So, no. It was like the first time I was stuck in such a situation. And, yeah. I didn't know what to do. What I did in the end is I agreed to do it. I did it only once and I didn't want to disappoint my friends. And they were into it. They've done it before. before. So yeah, I did it. I jumped.'
Past Dominant Emo : neutral
Vector Distance   : 0.275 (MAE)
   ⚠️ Memory vs Present Events Contradiction Detected (Threshold > 0.2)
 > Text splitted to sentences.
['"I sense that you\'re reflecting on this experience in a different way now compared to when it first happened. What\'s changed for you since then? Is there something specific that\'s making you feel more uneasy or anxious about the experience now?"']
 > Processing time: 7.560837984085083
```

### the llm doesn't understand that i am refering to the same story

```

🟢 TURN 1 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding face modality (weak neutral: 0.54)
Discarding audio modality (confidence too far from best: diff=0.36)

Past retrieved memories: 
 {'past_text': None, 'mae': 0.0, 'is_contradiction': False, 'past_top': None}

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I want to tell you about an experience I had one year ago with my friends.'

💬 Agent: That sounds like a fun memory! What made that particular experience stand out from others you've had with your friends?

📖 TEXT MODALITY:
   joy: 0.89
   neutral: 0.06
   surprise: 0.03

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.08
   fear: 0.03
   joy: 0.01
   sadness: 0.53
   surprise: 0.02
   neutral: 0.31

🎭 VIDEO MODALITY:
   neutral: 0.54
   sadness: 0.29
   anger: 0.11

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.84)
Audio confident : True (diff=0.22)
Face confident  : True (diff=0.25)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: joy (confidence=0.89)
 > Text splitted to sentences.
['That sounds like a fun memory!', "What made that particular experience stand out from others you've had with your friends?"]
 > Processing time: 3.896897077560425
 > Real-time factor: 0.4870679561956249
💾 Turn stored in Chroma (id: I want to tell you about an experience I...)
Stored FINAL emotion in Chroma → joy ({'joy': 0.8940457701683044, 'neutral': 0.055774204432964325, 'surprise': 0.03160466253757477, 'sadness': 0.006993794348090887, 'fear': 0.0054436796344816685, 'disgust': 0.003673579776659608, 'anger': 0.0024643372744321823})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 4.13 seconds
  - RoBERTa (Text Emotion)   : 0.83 seconds
  - Audeering (Audio Emotion): 0.67 seconds
  - DeepFace (Video Emotion) : 1.63 seconds (10 frames processed)
  - ChromaDB (Memory Fetch)  : 0.53 seconds
  - LLM Generation           : 13.08 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 20.88 seconds
============================================================


------------------------------------------------------------
🟢 TURN 2 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.54)
Discarding face modality (weak neutral: 0.66)

Past retrieved memories: 
 {'past_text': 'I want to tell you about an experience I had one year ago with my friends.', 'mae': 0.21721608407928475, 'is_contradiction': True, 'past_top': 'joy'}

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'So it's an unforgettable, amazing, totally awesome experience which made me so happy even though I didn't expect it at all. I was just terrified when my friends proposed me that.'

💬 Agent: It seems like your emotions are taking a different route this time compared to the past. What do you think has changed or is making you feel a sense of uncertainty now, given how you felt back then?

📖 TEXT MODALITY:
   fear: 0.75
   joy: 0.19
   surprise: 0.04

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.09
   fear: 0.03
   joy: 0.01
   sadness: 0.27
   surprise: 0.03
   neutral: 0.54

🎭 VIDEO MODALITY:
   neutral: 0.66
   surprise: 0.13
   sadness: 0.08

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.57)
Audio confident : True (diff=0.28)
Face confident  : True (diff=0.53)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: fear (confidence=0.75)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'I want to tell you about an experience I had one year ago with my friends.'
Past Dominant Emo : joy
Vector Distance   : 0.217 (MAE)
   ⚠️ Memory vs Present Events Contradiction Detected (Threshold > 0.2)
 > Text splitted to sentences.
['It seems like your emotions are taking a different route this time compared to the past.', 'What do you think has changed or is making you feel a sense of uncertainty now, given how you felt back then?']
 > Processing time: 6.929930925369263
 > Real-time factor: 0.5221602545940139
💾 Turn stored in Chroma (id: So it's an unforgettable, amazing, total...)
Stored FINAL emotion in Chroma → fear ({'fear': 0.7524601817131042, 'joy': 0.18593309819698334, 'surprise': 0.04484444856643677, 'neutral': 0.0100022591650486, 'sadness': 0.003132699755951762, 'anger': 0.002266273833811283, 'disgust': 0.0013610546011477709})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 4.52 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.39 seconds
  - DeepFace (Video Emotion) : 2.38 seconds (20 frames processed)
  - ChromaDB (Memory Fetch)  : 0.29 seconds
  - LLM Generation           : 33.11 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 40.71 seconds
============================================================


```

```

------------------------------------------------------------
🟢 TURN 3 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.59)
Discarding face modality (weak neutral: 0.40)

Past retrieved memories: 
 {'past_text': "So it's an unforgettable, amazing, totally awesome experience which made me so happy even though I didn't expect it at all. I was just terrified when my friends proposed me that.", 'mae': 0.06578344475877072, 'is_contradiction': False, 'past_top': 'fear'}

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'So I'm not uncertain about the emotions right now, like it was a great experience. I was just saying that in the beginning I thought it's going to be terrifying and I was terrified. Yeah, it just turned out to be great.'

💬 Agent: It sounds like your nervous energy from the start of this experience has carried over into this moment, echoing what you felt previously. How are you handling the lingering feeling of apprehension now that the outcome is so positive?

📖 TEXT MODALITY:
   fear: 0.98
   surprise: 0.01
   neutral: 0.01

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.08
   fear: 0.03
   joy: 0.01
   sadness: 0.21
   surprise: 0.03
   neutral: 0.59

🎭 VIDEO MODALITY:
   neutral: 0.40
   fear: 0.17
   anger: 0.14

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.98)
Audio confident : True (diff=0.38)
Face confident  : True (diff=0.24)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: fear (confidence=0.98)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'So it's an unforgettable, amazing, totally awesome experience which made me so happy even though I didn't expect it at all. I was just terrified when my friends proposed me that.'
Past Dominant Emo : fear
Vector Distance   : 0.066 (MAE)
   ✅ Memory & Present Events Alignment Detected (Threshold <= 0.2)
 > Text splitted to sentences.
['It sounds like your nervous energy from the start of this experience has carried over into this moment, echoing what you felt previously.', 'How are you handling the lingering feeling of apprehension now that the outcome is so positive?']
 > Processing time: 7.618186950683594
 > Real-time factor: 0.5019993253997718

🔄 [Semantic Memory] Background thread summarizing recent turns...
💾 Turn stored in Chroma (id: So I'm not uncertain about the emotions ...)
Stored FINAL emotion in Chroma → fear ({'fear': 0.982563316822052, 'surprise': 0.005875806789845228, 'neutral': 0.00516311964020133, 'joy': 0.0025564352981746197, 'disgust': 0.0014999391278252006, 'anger': 0.001243589329533279, 'sadness': 0.001097734784707427})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 5.00 seconds
  - RoBERTa (Text Emotion)   : 0.02 seconds
  - Audeering (Audio Emotion): 0.34 seconds
  - DeepFace (Video Emotion) : 2.00 seconds (18 frames processed)
  - ChromaDB (Memory Fetch)  : 0.38 seconds
  - LLM Generation           : 44.64 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 52.38 seconds
============================================================


------------------------------------------------------------

```





### here again it says that it makes a stringe impact than before...

```

 SYSTEM READY. Awaiting your turn.
============================================================


------------------------------------------------------------
🟢 TURN 1 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.65)

Past retrieved memories: 
 {'past_text': None, 'mae': 0.0, 'is_contradiction': False, 'past_top': None}

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'Hello, I want to tell you about an experience from my past.'

💬 Agent: It sounds like you're interested in sharing a significant memory with me. Can you start by telling me what made this experience stand out to you?

📖 TEXT MODALITY:
   neutral: 0.87
   joy: 0.05
   surprise: 0.03

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.07
   fear: 0.03
   joy: 0.02
   sadness: 0.17
   surprise: 0.03
   neutral: 0.65

🎭 VIDEO MODALITY:
   neutral: 0.36
   sadness: 0.22
   fear: 0.18

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.81)
Audio confident : True (diff=0.48)
Face confident  : False (diff=0.14)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: neutral (confidence=0.87)
 > Text splitted to sentences.
["It sounds like you're interested in sharing a significant memory with me.", 'Can you start by telling me what made this experience stand out to you?']
 > Processing time: 4.455049991607666
 > Real-time factor: 0.5002131146883099
💾 Turn stored in Chroma (id: Hello, I want to tell you about an exper...)
Stored FINAL emotion in Chroma → neutral ({'neutral': 0.8657591342926025, 'joy': 0.05361635982990265, 'surprise': 0.02910633571445942, 'sadness': 0.019930031150579453, 'fear': 0.015747694298624992, 'disgust': 0.011367762461304665, 'anger': 0.004472723230719566})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.00 seconds
  - RoBERTa (Text Emotion)   : 1.09 seconds
  - Audeering (Audio Emotion): 0.69 seconds
  - DeepFace (Video Emotion) : 0.97 seconds (4 frames processed)
  - ChromaDB (Memory Fetch)  : 0.43 seconds
  - LLM Generation           : 28.14 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 34.32 seconds
============================================================


------------------------------------------------------------
🟢 TURN 2 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding face modality (weak neutral: 0.47)

Past retrieved memories: 
 {'past_text': 'Hello, I want to tell you about an experience from my past.', 'mae': 0.2718211380748211, 'is_contradiction': True, 'past_top': 'neutral'}

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'So it's just an unforgettable experience for me. It was great really even though at the beginning I was very hesitant to actually do it. I was scared.'

💬 Agent: You're sharing a story that still gives you goosebumps! It seems like, in this instance, your initial apprehension didn't stop you from taking the leap and experiencing something truly wonderful. What is it about this experience that's making its impact on you even stronger than before?

📖 TEXT MODALITY:
   fear: 0.97
   neutral: 0.01
   joy: 0.01

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.09
   fear: 0.03
   joy: 0.01
   sadness: 0.40
   surprise: 0.02
   neutral: 0.42

🎭 VIDEO MODALITY:
   neutral: 0.47
   sadness: 0.27
   anger: 0.13

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.96)
Audio confident : False (diff=0.02)
Face confident  : True (diff=0.20)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: fear (confidence=0.97)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'Hello, I want to tell you about an experience from my past.'
Past Dominant Emo : neutral
Vector Distance   : 0.272 (MAE)
   ⚠️ Memory vs Present Events Contradiction Detected (Threshold > 0.2)
 > Text splitted to sentences.
["You're sharing a story that still gives you goosebumps!", "It seems like, in this instance, your initial apprehension didn't stop you from taking the leap and experiencing something truly wonderful.", "What is it about this experience that's making its impact on you even stronger than before?"]
 > Processing time: 9.22014307975769
 > Real-time factor: 0.4856671513890252
💾 Turn stored in Chroma (id: So it's just an unforgettable experience...)
Stored FINAL emotion in Chroma → fear ({'fear': 0.9671216607093811, 'neutral': 0.01000218465924263, 'joy': 0.009902686811983585, 'surprise': 0.008459980599582195, 'sadness': 0.002027126494795084, 'disgust': 0.0015227041440084577, 'anger': 0.0009636638569645584})
```

### good example of confusing the timeline

```
🗣️ User Said: 'Yes, so I was celebrating my 25th birthday one year ago. So my friends wanted to give me a special gift. They booked for me a bungee jump. I was very hesitant at the beginning because I have a fear of heights. I was so stressed before I jumped.'

💬 Agent: It's interesting that you're recalling this experience now, and it seems like your emotions are running a bit differently than they did back then. What do you think is causing this sense of trepidation or uncertainty for you in the present moment?

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'So I want to tell you a story about an incredible spirit I had one year ago when my friends bought me a pair of the gifts.'
Past Dominant Emo : joy
Vector Distance   : 0.264 (MAE)
   ⚠️ Memory vs Present Events Contradiction Detected (Threshold > 0.2)

**So the agent thinks that I was first happy when the event happend, and right now I uncertain about it, eventhogh I was just telling the story.**

```
 AGENT RESPONSE
============================================================
🗣️ User Said: 'I don't feel uncertain about the event right now. I was just uncertain before I jumped. After I jumped, actually it was great as I'm saying. So right now I'm happy. I did it. I'm very happy. But before jumping I was just uncertain.'

💬 Agent: It makes total sense that you're feeling anxious or uneasy about reliving this experience again, especially given your previous response to a similar situation. Can you tell me what's helping you feel more grounded and at ease now compared to how you felt beforehand?

📖 TEXT MODALITY:
   fear: 0.97
   joy: 0.02
   neutral: 0.00

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.07
   fear: 0.03
   joy: 0.02
   sadness: 0.17
   surprise: 0.03
   neutral: 0.65

🎭 VIDEO MODALITY:
   neutral: 0.50
   sadness: 0.19
   anger: 0.13

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.94)
Audio confident : True (diff=0.48)
Face confident  : True (diff=0.31)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: fear (confidence=0.97)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'Yes, so I was celebrating my 25th birthday one year ago. So my friends wanted to give me a special gift. They booked for me a bungee jump. I was very hesitant at the beginning because I have a fear of heights. I was so stressed before I jumped.'
Past Dominant Emo : fear
Vector Distance   : 0.010 (MAE)
   ✅ Memory & Present Events Alignment Detected (Threshold <= 0.2)
 > Text splitted to sentences.
["It makes total sense that you're feeling anxious or uneasy about reliving this experience again, especially given your previous response to a similar situation.", "Can you tell me what's helping you feel more grounded and at ease now compared to how you felt beforehand?"]

```

### anothe trial roberta 28 emotions

```



🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.55)
Discarding face modality (weak neutral: 0.48)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'So it's an unforgettable, amazing, totally awesome experience which made me so happy even though I didn't expect it at all. I was just terrified and my friends proposed me that.'

💬 Agent: It sounds like you're reliving a moment that left a big smile on your face! Given how much happiness it brought you then, it's no wonder it still feels special to you now - what's the most surprising part of that experience that still stands out in your mind?

📖 TEXT MODALITY:
   joy: 0.79
   fear: 0.14
   surprise: 0.03

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.08
   fear: 0.03
   joy: 0.01
   sadness: 0.26
   surprise: 0.03
   neutral: 0.55

🎭 VIDEO MODALITY:
   neutral: 0.48
   surprise: 0.28
   anger: 0.11

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.65)
Audio confident : True (diff=0.29)
Face confident  : True (diff=0.20)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: joy (confidence=0.79)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'I want to tell you about an experience I had one year ago with my friends.'
Past Dominant Emo : joy
Vector Distance   : 0.082 (MAE)
   ✅ Memory & Present Events Alignment Detected (Threshold <= 0.2)
 > Text splitted to sentences.
["It sounds like you're reliving a moment that left a big smile on your face!", "Given how much happiness it brought you then, it's no wonder it still feels special to you now - what's the most surprising part of that experience that still stands out in your mind?"]
 > Processing time: 9.085227012634277
 > Real-time factor: 0.49088757456232307

 ```

 ### good conversation

 ```
 ============================================================
✅ SYSTEM READY. Awaiting your turn.
============================================================


------------------------------------------------------------
🟢 TURN 1 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.56)
Discarding face modality (weak neutral: 0.62)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I want to tell you about an experience I had one year ago with my friends.'

💬 Agent: That sounds like a great story! Can you tell me more about what made that experience so special?

📖 TEXT MODALITY:
   joy: 0.65
   neutral: 0.19
   surprise: 0.13

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.09
   fear: 0.03
   joy: 0.01
   sadness: 0.24
   surprise: 0.03
   neutral: 0.56

🎭 VIDEO MODALITY:
   neutral: 0.62
   surprise: 0.20
   sadness: 0.13

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.47)
Audio confident : True (diff=0.32)
Face confident  : True (diff=0.42)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: joy (confidence=0.65)
 > Text splitted to sentences.
['That sounds like a great story!', 'Can you tell me more about what made that experience so special?']
 > Processing time: 3.300955057144165
 > Real-time factor: 0.48925883933392156
💾 Turn stored in Chroma (id: I want to tell you about an experience I...)
Stored FINAL emotion in Chroma → joy ({'joy': 0.6546567274207427, 'neutral': 0.18856876830712982, 'surprise': 0.1341493457188443, 'sadness': 0.009929270883737469, 'anger': 0.008831267042187148, 'fear': 0.0026884713929834643, 'disgust': 0.0011761492343751263})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 3.95 seconds
  - RoBERTa (Text Emotion)   : 0.51 seconds
  - Audeering (Audio Emotion): 0.57 seconds
  - DeepFace (Video Emotion) : 1.32 seconds (8 frames processed)
  - ChromaDB (Memory Fetch)  : 0.27 seconds
  - LLM Generation           : 35.62 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 42.23 seconds
============================================================


------------------------------------------------------------
🟢 TURN 2 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.55)
Discarding face modality (weak neutral: 0.48)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'So it's an unforgettable, amazing, totally awesome experience which made me so happy even though I didn't expect it at all. I was just terrified and my friends proposed me that.'

💬 Agent: It sounds like you're reliving a moment that left a big smile on your face! Given how much happiness it brought you then, it's no wonder it still feels special to you now - what's the most surprising part of that experience that still stands out in your mind?

📖 TEXT MODALITY:
   joy: 0.79
   fear: 0.14
   surprise: 0.03

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.04
   disgust: 0.08
   fear: 0.03
   joy: 0.01
   sadness: 0.26
   surprise: 0.03
   neutral: 0.55

🎭 VIDEO MODALITY:
   neutral: 0.48
   surprise: 0.28
   anger: 0.11

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.65)
Audio confident : True (diff=0.29)
Face confident  : True (diff=0.20)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: joy (confidence=0.79)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'I want to tell you about an experience I had one year ago with my friends.'
Past Dominant Emo : joy
Vector Distance   : 0.082 (MAE)
   ✅ Memory & Present Events Alignment Detected (Threshold <= 0.2)
 > Text splitted to sentences.
["It sounds like you're reliving a moment that left a big smile on your face!", "Given how much happiness it brought you then, it's no wonder it still feels special to you now - what's the most surprising part of that experience that still stands out in your mind?"]
 > Processing time: 9.085227012634277
 > Real-time factor: 0.49088757456232307
💾 Turn stored in Chroma (id: So it's an unforgettable, amazing, total...)
Stored FINAL emotion in Chroma → joy ({'joy': 0.7932981609426399, 'fear': 0.14111211028558596, 'surprise': 0.03128237444360666, 'sadness': 0.018545152973993277, 'anger': 0.009705037299024376, 'neutral': 0.004072680731366594, 'disgust': 0.0019844833237831466})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 4.86 seconds
  - RoBERTa (Text Emotion)   : 0.31 seconds
  - Audeering (Audio Emotion): 0.37 seconds
  - DeepFace (Video Emotion) : 1.63 seconds (14 frames processed)
  - ChromaDB (Memory Fetch)  : 0.38 seconds
  - LLM Generation           : 37.12 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 44.67 seconds
============================================================


------------------------------------------------------------
🟢 TURN 3 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.58)
Discarding face modality (weak neutral: 0.57)

🧠 Sending profile to LLM...

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'What surprised me a lot was that... was how nice this experience was. Because I was very hesitant at the beginning. I have a fear of heights, fear of heights, and I didn't want to do it. I did it because my friends bought it for me. Bought this, gave this experience a boost. Bunny jumping for me. So that's the only reason for which I jumped. I wouldn't do it. just for myself, I was too scared but it turned out great and that's the most surprising and nice part which I want to share with you'

💬 Agent: It's amazing how certain experiences can still make us feel like we're walking on air, even years later! It sounds like this experience was a perfect blend of your friends' encouragement and the thrill of facing your fears - what do you think it is about that day that makes it so special to you now?

📖 TEXT MODALITY:
   joy: 0.59
   surprise: 0.21
   fear: 0.15

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.08
   fear: 0.03
   joy: 0.01
   sadness: 0.23
   surprise: 0.03
   neutral: 0.58

🎭 VIDEO MODALITY:
   neutral: 0.57
   sadness: 0.18
   anger: 0.10

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.38)
Audio confident : True (diff=0.35)
Face confident  : True (diff=0.39)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: joy (confidence=0.59)

🕰️ TEMPORAL MEMORY CHECK
Past Event Matched: 'So it's an unforgettable, amazing, totally awesome experience which made me so happy even though I didn't expect it at all. I was just terrified and my friends proposed me that.'
Past Dominant Emo : joy
Vector Distance   : 0.058 (MAE)
   ✅ Memory & Present Events Alignment Detected (Threshold <= 0.2)
 > Text splitted to sentences.
["It's amazing how certain experiences can still make us feel like we're walking on air, even years later!", "It sounds like this experience was a perfect blend of your friends' encouragement and the thrill of facing your fears - what do you think it is about that day that makes it so special to you now?"]
 > Processing time: 9.591722965240479
 > Real-time factor: 0.48310040243666524

🔄 [Semantic Memory] Background thread summarizing recent turns...
💾 Turn stored in Chroma (id: What surprised me a lot was that... was ...)
Stored FINAL emotion in Chroma → joy ({'joy': 0.5904369958774363, 'surprise': 0.20579810385438346, 'fear': 0.14804400477990667, 'sadness': 0.028835394885861693, 'anger': 0.014274084278717257, 'neutral': 0.009473867335145788, 'disgust': 0.003137548988548819})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 18.03 seconds
  - RoBERTa (Text Emotion)   : 0.09 seconds
  - Audeering (Audio Emotion): 0.49 seconds
  - DeepFace (Video Emotion) : 4.09 seconds (40 frames processed)
  - ChromaDB (Memory Fetch)  : 0.52 seconds
  - LLM Generation           : 66.57 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 89.78 seconds
============================================================


------------------------------------------------------------
🟢 TURN 4 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding face modality (weak neutral: 0.52)

🧠 Sending profile to LLM...

✅ [Semantic Memory] Running Summary Updated in Background! (Latency: 75.69 seconds)

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'There is something very similar that is going to happen, at least I hope so, because I'm going to try sailing for the first time the next week. Right now I'm very nervous, but I really hope it will end the same as this Banchi experience. So first, right now I'm nervous but I hope to be very satisfied and happy afterwards.'

💬 Agent: It sounds like you're looking forward to another exciting experience! It's understandable that you might feel a little nervous about trying sailing for the first time - what are some of your hopes or expectations for this new adventure?

📖 TEXT MODALITY:
   joy: 0.54
   fear: 0.33
   surprise: 0.05

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.07
   fear: 0.03
   joy: 0.01
   sadness: 0.35
   surprise: 0.03
   neutral: 0.48

🎭 VIDEO MODALITY:
   neutral: 0.52
   sadness: 0.19
   surprise: 0.16

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.21)
Audio confident : False (diff=0.13)
Face confident  : True (diff=0.32)

🧠 DECISION DEBUG
Decision type: full_agreement
Number of confident modalities: 1
 - text: joy (confidence=0.54)
 > Text splitted to sentences.
["It sounds like you're looking forward to another exciting experience!", "It's understandable that you might feel a little nervous about trying sailing for the first time - what are some of your hopes or expectations for this new adventure?"]
 > Processing time: 6.947097301483154
 > Real-time factor: 0.4912246520577975
💾 Turn stored in Chroma (id: There is something very similar that is ...)
Stored FINAL emotion in Chroma → joy ({'joy': 0.5429146924032616, 'fear': 0.33024677493729737, 'surprise': 0.05430026486793625, 'sadness': 0.03670667470071324, 'neutral': 0.021181796360971866, 'anger': 0.013272180030925968, 'disgust': 0.0013776166988937381})

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 10.09 seconds
  - RoBERTa (Text Emotion)   : 0.51 seconds
  - Audeering (Audio Emotion): 0.60 seconds
  - DeepFace (Video Emotion) : 4.61 seconds (30 frames processed)
  - ChromaDB (Memory Fetch)  : 0.38 seconds
  - LLM Generation           : 173.27 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 189.46 seconds
============================================================


------------------------------------------------------------
🟢 TURN 5 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
Discarding audio modality (weak neutral: 0.61)

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
* The user had an unforgettable experience with their friends one year ago.
* The experience involved bunny jumping, which was surprising given the user's fear of heights.
* The user was hesitant at first but did the activity because their friends encouraged them to do so.
* The experience turned out great and made the user very happy.

2. Emotional Trajectory:
* The user's mood appears to be nostalgic and happy as they relive a positive memory from the past.
* The user's core struggle seems to be about facing fears and overcoming initial hesitation, with their friends' support playing a significant role in making the experience special.

        [Hidden Context for Agent]

        User message: "I hope that I will overcome the fear I have towards water. So it will be similar to the experience I had with bandit jumping in which I overcame the fear I had for heights."
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
        

============================================================
🤖 AGENT RESPONSE
============================================================
🗣️ User Said: 'I hope that I will overcome the fear I have towards water. So it will be similar to the experience I had with bandit jumping in which I overcame the fear I had for heights.'

💬 Agent: It's interesting that you're hoping for a similar experience with sailing, but I sense that your emotions right now might be a bit mixed. You seem to be feeling joyful about the prospect of overcoming your fear and having another exciting moment, but at the same time, there's a hint of neutral calmness - like you're still processing everything. How are you truly feeling underneath all this excitement?

📖 TEXT MODALITY:
   joy: 0.89
   neutral: 0.04
   surprise: 0.03

🎵 AUDIO MODALITY:
   Ekman probabilities:
   anger: 0.03
   disgust: 0.07
   fear: 0.03
   joy: 0.02
   sadness: 0.20
   surprise: 0.03
   neutral: 0.61

🎭 VIDEO MODALITY:
   neutral: 0.87
   sadness: 0.06
   joy: 0.05

🔎 CONFIDENCE CHECK
Text confident  : True (diff=0.85)
Audio confident : True (diff=0.41)
Face confident  : True (diff=0.81)

🧠 DECISION DEBUG
Decision type: conflict
Number of confident modalities: 2
 - text: joy (confidence=0.89)
 - face: neutral (confidence=0.87)
 > Text splitted to sentences.
["It's interesting that you're hoping for a similar experience with sailing, but I sense that your emotions right now might be a bit mixed.", "You seem to be feeling joyful about the prospect of overcoming your fear and having another exciting moment, but at the same time, there's a hint of neutral calmness - like you're still processing everything.", 'How are you truly feeling underneath all this excitement?']
 > Processing time: 12.671913862228394
 > Real-time factor: 0.5062025360556395
Skipping DB storage (no resolved emotion)

============================================================
⏱️ LATENCY BENCHMARKING REPORT
============================================================
  - Whisper (Speech to Text) : 6.97 seconds
  - RoBERTa (Text Emotion)   : 0.67 seconds
  - Audeering (Audio Emotion): 0.91 seconds
  - DeepFace (Video Emotion) : 2.90 seconds (23 frames processed)
  - ChromaDB (Memory Fetch)  : 0.00 seconds
  - LLM Generation           : 223.61 seconds
  -------------------------------------------
  - TOTAL PIPELINE LATENCY   : 235.07 seconds
============================================================


------------------------------------------------------------
🟢 TURN 6 | Press [ENTER] to start speaking (or type 'q' to quit): 

🔴 Recording! Speak naturally.
🛑 Press [ENTER] when you are finished talking...


✅ Recording stopped.

Processing Turn... Please wait.
🧠 Resolving previous emotional conflict/no_data from user reply...
{'final_emotion': 'joy', 'emotion_distribution': {'joy': 0.8240514860997153, 'surprise': 0.09220321621801414, 'neutral': 0.04913509366201731, 'anger': 0.015894527757168625, 'fear': 0.010445368631916285, 'sadness': 0.006752746331815024, 'disgust': 0.0015175612993532604}}
💾 Stored resolved emotion → joy ({'joy': 0.8240514860997153, 'surprise': 0.09220321621801414, 'neutral': 0.04913509366201731, 'anger': 0.015894527757168625, 'fear': 0.010445368631916285, 'sadness': 0.006752746331815024, 'disgust': 0.0015175612993532604})

🧠 Sending profile to LLM...

🧠 Current Semantic Summary:
  1. Core Facts & Context:
* The user had an unforgettable experience with their friends one year ago.
* The experience involved bunny jumping, which was surprising given the user's fear of heights.
* The user was hesitant at first but did the activity because their friends encouraged them to do so.
* The experience turned out great and made the user very happy.

2. Emotional Trajectory:
* The user's mood appears to be nostalgic and happy as they relive a positive memory from the past.
* The user's core struggle seems to be about facing fears and overcoming initial hesitation, with their friends' support playing a significant role in making the experience special.
```