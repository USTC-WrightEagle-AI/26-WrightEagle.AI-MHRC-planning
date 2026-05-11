# RoboCup@Home Speech Model Test Report

**Generated:** 2026-04-11 00:15:58
**Environment:** CPU | Windows | sherpa-onnx Python API

## Table of Contents

- [1. Overview](#1-overview)
- [2. Model List](#2-model-list)
- [3. Test Texts](#3-test-texts)
- [4. TTS Results](#4-tts-results)
- [5. Offline ASR Results](#5-offline-asr-results)
- [6. Streaming ASR Results](#6-streaming-asr-results)
- [7. Performance Comparison](#7-performance-comparison)
- [8. Audio Files](#8-audio-files)
- [9. Conclusions](#9-conclusions)

## 1. Overview

| Item | Description |
|------|-------------|
| Goal | Evaluate speech models for RoboCup@Home Task 1 |
| Scope | TTS + Offline ASR (5) + Online ASR (2) |
| Hardware | CPU only |
| Date | 2026-04-11 00:13:12 |

## 2. Model List

### 2.1 TTS Model

| Model | Type | Language | Notes |
|-------|------|----------|-------|
| Kokoro int8 v0.19 | Offline TTS | English | Multi-voice support |

### 2.2 Offline ASR Models

| Model | Architecture | Language | Features |
|-------|-------------|----------|----------|
| Whisper Small EN | Transformer Enc-Dec | English | OpenAI, robust |
| Moonshine Tiny Quantized | E2E | English | Lightweight, edge-friendly |
| FireRed ASR2 | Paraformer-like | ZH+EN | Bilingual support |
| MedaSR CTC EN Int8 | NeMo CTC | English | Medical domain optimized |
| Qwen3-ASR 0.6B Int8 | LLM-based ASR | ZH+EN | Large language model based |

### 2.3 Streaming ASR Models

| Model | Architecture | Language | Latency |
|-------|-------------|----------|--------|
| NeMo FastConformer CTC 80ms | CTC Streaming | English | 80ms chunk |
| Streaming Zipformer Kroko | Transducer | English | Real-time |

## 3. Test Texts

**Total: 16 texts** (General: 8 + RoboCup Commands: 8, all English)

**General Conversation:**

1. Hello, how are you doing today? I hope everything is going well for you.
2. The weather is really nice today, perfect for a walk outside.
3. I had a great lunch with my friends at the new restaurant downtown.
4. Could you please help me find my keys? I think I left them on the table.
5. The meeting has been rescheduled to three o'clock tomorrow afternoon.
6. I am learning about artificial intelligence and machine learning these days.
7. My favorite color is blue, but I also like green and yellow quite a lot.
8. She went to the supermarket to buy some milk, bread, and fresh fruit.

**RoboCup@Home Commands:**

9. Bring me a coke from the kitchen.
10. Go to the living room and wait there.
11. Point to the person wearing a red shirt.
12. Follow me to the bedroom please.
13. What objects do you see on the dining table?
14. Pick up the green cup and put it on the shelf.
15. Tell me the name of this object you are holding.
16. Navigate to the door and open it for me.

## 4. TTS Results

- **Model:** `kokoro-int8-en-v0_19`
- **Speaker ID:** `10`
- **Generated:** `16/16` files
- **Total Duration:** `46.487s`
- **Total Time:** `71.679s`
- **Avg RTF:** `1.542`

**Details:**

| # | File | Text Preview | Duration(s) | RTF |
|---|------|-------------|------------|-----|
| 0 | tts_kokoro_s10_00.wav | Hello, how are you doing today? I hope everyt... | 3.753 | 1.705 |
| 1 | tts_kokoro_s10_01.wav | The weather is really nice today, perfect for... | 3.296 | 1.48 |
| 2 | tts_kokoro_s10_02.wav | I had a great lunch with my friends at the ne... | 3.478 | 1.497 |
| 3 | tts_kokoro_s10_03.wav | Could you please help me find my keys? I thin... | 3.765 | 1.563 |
| 4 | tts_kokoro_s10_04.wav | The meeting has been rescheduled to three o'c... | 3.683 | 1.425 |
| 5 | tts_kokoro_s10_05.wav | I am learning about artificial intelligence a... | 4.122 | 1.406 |
| 6 | tts_kokoro_s10_06.wav | My favorite color is blue, but I also like gr... | 4.344 | 1.42 |
| 7 | tts_kokoro_s10_07.wav | She went to the supermarket to buy some milk,... | 3.719 | 1.486 |
| 8 | tts_kokoro_s10_08.wav | Bring me a coke from the kitchen. | 1.556 | 1.679 |
| 9 | tts_kokoro_s10_09.wav | Go to the living room and wait there. | 1.88 | 1.726 |
| 10 | tts_kokoro_s10_10.wav | Point to the person wearing a red shirt. | 2.052 | 1.59 |
| 11 | tts_kokoro_s10_11.wav | Follow me to the bedroom please. | 1.721 | 1.626 |
| 12 | tts_kokoro_s10_12.wav | What objects do you see on the dining table? | 2.432 | 1.498 |
| 13 | tts_kokoro_s10_13.wav | Pick up the green cup and put it on the shelf... | 2.187 | 1.73 |
| 14 | tts_kokoro_s10_14.wav | Tell me the name of this object you are holdi... | 2.329 | 1.65 |
| 15 | tts_kokoro_s10_15.wav | Navigate to the door and open it for me. | 2.172 | 1.563 |

## 5. Offline ASR Results

### 5.1 Summary

| Model | Char Accuracy | Exact Match | RTF | Status |
|-------|-------------|-------------|-----|--------|
| whisper | 93.3% | 81.2% | 0.563 | OK |
| moonshine | 89.4% | 75.0% | 0.025 | OK |
| fire_red | 39.6% | 0.0% | 0.363 | OK |
| medasr | 68.2% | 31.2% | 0.034 | OK |
| qwen3 | 88.1% | 75.0% | 0.429 | OK |

### 5.2 WHISPER Details

| # | Ground Truth | Recognized | Match | Accuracy |
|---|-------------|------------|-------|----------|
| 1 | Hello, how are you doing today? I hope everyt | Hello, how are you doing today? I hope everyt | Y | - |
| 2 | The weather is really nice today, perfect for | The weather is really nice today, perfect for | Y | - |
| 3 | I had a great lunch with my friends at the ne | I had a great lunch with my friends at the ne | Y | - |
| 4 | Could you please help me find my keys? I thin | Could you please help me find my keys? I thin | Y | - |
| 5 | The meeting has been rescheduled to three o'c | The meeting has been rescheduled to 3 o'clock | N | - |
| 6 | I am learning about artificial intelligence a | I am learning about artificial intelligence a | Y | - |
| 7 | My favorite color is blue, but I also like gr | My favorite color is blue, but I also like gr | Y | - |
| 8 | She went to the supermarket to buy some milk, | She went to the supermarket to buy some milk, | N | - |
| 9 | Bring me a coke from the kitchen. | Bring me a coke from the kitchen. | Y | - |
| 10 | Go to the living room and wait there. | Go to the living room and wait there. | Y | - |
| 11 | Point to the person wearing a red shirt. | Point to the person wearing a red shirt. | Y | - |
| 12 | Follow me to the bedroom please. | Follow me to the bedroom, please. | N | - |
| 13 | What objects do you see on the dining table? | What objects do you see on the dining table? | Y | - |
| 14 | Pick up the green cup and put it on the shelf | Pick up the green cup and put it on the shelf | Y | - |
| 15 | Tell me the name of this object you are holdi | Tell me the name of this object you are holdi | Y | - |
| 16 | Navigate to the door and open it for me. | navigate to the door and open it for me. | Y | - |


### 5.2 MOONSHINE Details

| # | Ground Truth | Recognized | Match | Accuracy |
|---|-------------|------------|-------|----------|
| 1 | Hello, how are you doing today? I hope everyt | Hello, how are you doing today? I hope everyt | Y | - |
| 2 | The weather is really nice today, perfect for | The weather is really nice today, perfect for | Y | - |
| 3 | I had a great lunch with my friends at the ne | I had a great lunch with my friends at the ne | Y | - |
| 4 | Could you please help me find my keys? I thin | Could you please help me find my keys? I thin | Y | - |
| 5 | The meeting has been rescheduled to three o'c | The meeting has been rescheduled to three o'c | Y | - |
| 6 | I am learning about artificial intelligence a | I am learning about artificial intelligence a | Y | - |
| 7 | My favorite color is blue, but I also like gr | My favourite colour is blue, but I also like  | N | - |
| 8 | She went to the supermarket to buy some milk, | She went to the supermarket to buy some milk, | N | - |
| 9 | Bring me a coke from the kitchen. | Bring me a Coke from the kitchen. | Y | - |
| 10 | Go to the living room and wait there. | Go to the living room and wait there. | Y | - |
| 11 | Point to the person wearing a red shirt. | Point to the person wearing a red shirt | N | - |
| 12 | Follow me to the bedroom please. | Follow me to the bedroom, please. | N | - |
| 13 | What objects do you see on the dining table? | What objects do you see on the dining table? | Y | - |
| 14 | Pick up the green cup and put it on the shelf | Pick up the green cup and put it on the shelf | Y | - |
| 15 | Tell me the name of this object you are holdi | Tell me the name of this object you are holdi | Y | - |
| 16 | Navigate to the door and open it for me. | Navigate to the door and open it for me. | Y | - |


### 5.2 FIRE_RED Details

| # | Ground Truth | Recognized | Match | Accuracy |
|---|-------------|------------|-------|----------|
| 1 | Hello, how are you doing today? I hope everyt | HELL HOW ARE YOU DOING TODAY I HOPE EVERYTHIN | N | - |
| 2 | The weather is really nice today, perfect for | WEATHER IS REALLY NICE TODAY PERFECT FOR A WA | N | - |
| 3 | I had a great lunch with my friends at the ne | HAD A GREAT LUNCH WITH MY FRIENDS AT THE NEW  | N | - |
| 4 | Could you please help me find my keys? I thin | YOU PLEASE HELP ME FIND MY KE I THINK I LEFT  | N | - |
| 5 | The meeting has been rescheduled to three o'c | THE MEETING HAS BEEN RESCHEDULED TO THREE O'C | N | - |
| 6 | I am learning about artificial intelligence a | I AMARNING ABOUT ARIFICIAL INTELLIGENCE AND M | N | - |
| 7 | My favorite color is blue, but I also like gr | MY FAVORITE COLOR IS BLUE BUT I ALSO LIKE GRE | N | - |
| 8 | She went to the supermarket to buy some milk, | SHE WENT TO THE SUPER MCKET TO BUY SOME MILK  | N | - |
| 9 | Bring me a coke from the kitchen. | ING A COKE FROM THE KITHE | N | - |
| 10 | Go to the living room and wait there. | GO TO THE LIVING ROOM AND WAIT THERE | N | - |
| 11 | Point to the person wearing a red shirt. | POINT TO THE PERSON WEARING A RED SHIRT | N | - |
| 12 | Follow me to the bedroom please. | FOLLOW ME TO THE BEDROOM PLEASE | N | - |
| 13 | What objects do you see on the dining table? | WHAT OBJECT DO YOU SEE ON THE DINING TABLE | N | - |
| 14 | Pick up the green cup and put it on the shelf | ICK UP THE GREENUP AND PUT IT ON THE SHELF | N | - |
| 15 | Tell me the name of this object you are holdi | TELL ME THE NAME OF THIS OBJECT YOU ARE HOLDI | N | - |
| 16 | Navigate to the door and open it for me. | NAVIGATE TO THE DOOR AND OPEN IT FOR ME | N | - |


### 5.2 MEDASR Details

| # | Ground Truth | Recognized | Match | Accuracy |
|---|-------------|------------|-------|----------|
| 1 | Hello, how are you doing today? I hope everyt | Hello, how are you doing today. I hope everyt | N | - |
| 2 | The weather is really nice today, perfect for | the weather is really nice today, perfect for | Y | - |
| 3 | I had a great lunch with my friends at the ne | I had a great lunch with my friends at the ne | Y | - |
| 4 | Could you please help me find my keys? I thin | Could you please help me find my keys. I thin | N | - |
| 5 | The meeting has been rescheduled to three o'c | aame] meeting has been rescheduled to 3 o'clo | N | - |
| 6 | I am learning about artificial intelligence a | I am learning about artificial intelligence a | Y | - |
| 7 | My favorite color is blue, but I also like gr | My favorite color is blue, but I also like gr | Y | - |
| 8 | She went to the supermarket to buy some milk, | She went to the supermoket to buy some milk,  | N | - |
| 9 | Bring me a coke from the kitchen. | aName] bring me a co from the kitchen. | N | - |
| 10 | Go to the living room and wait there. | go to the living room and wait there. | Y | - |
| 11 | Point to the person wearing a red shirt. | ] poting to the person wearing a w reed. | N | - |
| 12 | Follow me to the bedroom please. | a] follow me to the bedroom please. | N | - |
| 13 | What objects do you see on the dining table? | what objects do see on the dining table. | N | - |
| 14 | Pick up the green cup and put it on the shelf | pick up the green cut and put it on the shelf | N | - |
| 15 | Tell me the name of this object you are holdi | tell me the name of this object.You are holdi | N | - |
| 16 | Navigate to the door and open it for me. | e] nagate to the door and open it for me. | N | - |


### 5.2 QWEN3 Details

| # | Ground Truth | Recognized | Match | Accuracy |
|---|-------------|------------|-------|----------|
| 1 | Hello, how are you doing today? I hope everyt | Hello, how are you doing today? I hope everyt | Y | - |
| 2 | The weather is really nice today, perfect for | The weather is really nice today. Perfect for | N | - |
| 3 | I had a great lunch with my friends at the ne | I had a great lunch with my friends at the ne | Y | - |
| 4 | Could you please help me find my keys? I thin | Could you please help me find my keys? I thin | Y | - |
| 5 | The meeting has been rescheduled to three o'c | The meeting has been rescheduled to three o'c | Y | - |
| 6 | I am learning about artificial intelligence a | I am learning about artificial intelligence—a | N | - |
| 7 | My favorite color is blue, but I also like gr | My favourite colour is blue, but I also like  | N | - |
| 8 | She went to the supermarket to buy some milk, | She went to the supermarket to buy some milk, | Y | - |
| 9 | Bring me a coke from the kitchen. | Bring me a coke from the kitchen. | Y | - |
| 10 | Go to the living room and wait there. | Go to the living room and wait there. | Y | - |
| 11 | Point to the person wearing a red shirt. | Point to the person wearing a red shirt. | Y | - |
| 12 | Follow me to the bedroom please. | Follow me to the bedroom, please. | N | - |
| 13 | What objects do you see on the dining table? | What objects do you see on the dining table? | Y | - |
| 14 | Pick up the green cup and put it on the shelf | Pick up the green cup and put it on the shelf | Y | - |
| 15 | Tell me the name of this object you are holdi | Tell me the name of this object you are holdi | Y | - |
| 16 | Navigate to the door and open it for me. | Navigate to the door and open it for me. | Y | - |



## 6. Streaming ASR Results

### 6.1 Summary

| Model | Char Accuracy | Exact Match | RTF | Status |
|-------|-------------|-------------|-----|--------|
| nemo_streaming | 68.8% | 0.0% | 0.240 | OK |
| streaming_zipformer | 88.5% | 0.0% | 0.035 | OK |

### 6.2 NEMO_STREAMING Details

| # | Ground Truth | Recognized | Match |
|---|-------------|------------|-------|
| 1 | Hello, how are you doing today? I hope everyt | allow how are you doing today i hope everythi | N |
| 2 | The weather is really nice today, perfect for | the weather is really nice today perfect for  | N |
| 3 | I had a great lunch with my friends at the ne | i had a great lunch with my friends at the ne | N |
| 4 | Could you please help me find my keys? I thin | could you please help me find my keys i think | N |
| 5 | The meeting has been rescheduled to three o'c | the meeting has been recheduled to three o'cl | N |
| 6 | I am learning about artificial intelligence a | i am learning about artificial intelligence a | N |
| 7 | My favorite color is blue, but I also like gr | my favourite colour is blue but i also like g | N |
| 8 | She went to the supermarket to buy some milk, | she went to the supermarket to buy some milk  | N |
| 9 | Bring me a coke from the kitchen. | bring me a coke from the kitchen | N |
| 10 | Go to the living room and wait there. | go to the living room and wait there | N |
| 11 | Point to the person wearing a red shirt. | point to the person wearing a red shirt | N |
| 12 | Follow me to the bedroom please. | follow me to the bedroom please | N |
| 13 | What objects do you see on the dining table? | what objects do you see on the dining table | N |
| 14 | Pick up the green cup and put it on the shelf | pick up the green cut and put it on the shelf | N |
| 15 | Tell me the name of this object you are holdi | tell me the name of this object you are holdi | N |
| 16 | Navigate to the door and open it for me. | navigate to the door and open it for me | N |

### 6.2 STREAMING_ZIPFORMER Details

| # | Ground Truth | Recognized | Match |
|---|-------------|------------|-------|
| 1 | Hello, how are you doing today? I hope everyt | Hello, how are you doing today? I hope everyt | N |
| 2 | The weather is really nice today, perfect for | the weather is really nice today, perfect for | N |
| 3 | I had a great lunch with my friends at the ne | I had a great lunch with my friends at the ne | N |
| 4 | Could you please help me find my keys? I thin | Could you please help me find my keys? I thin | N |
| 5 | The meeting has been rescheduled to three o'c | the meeting has been rescheduled to three o'c | N |
| 6 | I am learning about artificial intelligence a | I am learning about artificial intelligence a | N |
| 7 | My favorite color is blue, but I also like gr | My favorite color is blue, but I also like gr | N |
| 8 | She went to the supermarket to buy some milk, | she went to the supermarket to buy some milk, | N |
| 9 | Bring me a coke from the kitchen. | Bring me a coke from | N |
| 10 | Go to the living room and wait there. | go to the living room | N |
| 11 | Point to the person wearing a red shirt. | point to the person wearing a red shirt | N |
| 12 | Follow me to the bedroom please. | Follow me to the bedro | N |
| 13 | What objects do you see on the dining table? | What objects do you see on the dining table | N |
| 14 | Pick up the green cup and put it on the shelf | Pick up the green cup and put it on the shelf | N |
| 15 | Tell me the name of this object you are holdi | tell me the name of this object you are holdi | N |
| 16 | Navigate to the door and open it for me. | navigate to the door and open it for me | N |


## 7. Performance Comparison

### 7.1 Ranked by Accuracy

| Rank | Model | Mode | Char Accuracy | Exact Match | RTF |
|------|-------|------|---------------|-------------|-----|
| 1 | whisper | Offline | 93.3% | 81.2% | 0.563 |
| 2 | moonshine | Offline | 89.4% | 75.0% | 0.025 |
| 3 | streaming_zipformer | Streaming | 88.5% | 0.0% | 0.035 |
| 4 | qwen3 | Offline | 88.1% | 75.0% | 0.429 |
| 5 | nemo_streaming | Streaming | 68.8% | 0.0% | 0.240 |
| 6 | medasr | Offline | 68.2% | 31.2% | 0.034 |
| 7 | fire_red | Offline | 39.6% | 0.0% | 0.363 |

### 7.2 Ranked by Speed (RTF)

| Rank | Model | Mode | RTF | Char Accuracy |
|------|-------|------|-----|--------------|
| 1 | moonshine | Offline | 0.025 | 89.4% |
| 2 | medasr | Offline | 0.034 | 68.2% |
| 3 | streaming_zipformer | Streaming | 0.035 | 88.5% |
| 4 | nemo_streaming | Streaming | 0.240 | 68.8% |
| 5 | fire_red | Offline | 0.363 | 39.6% |
| 6 | qwen3 | Offline | 0.429 | 88.1% |
| 7 | whisper | Offline | 0.563 | 93.3% |

### 7.3 Recommendations

- **Best Accuracy:** `whisper` (Offline) - Accuracy: **93.3%**
- **Fastest Speed:** `moonshine` (Offline) - RTF: **0.025**


## 8. Audio Files

All TTS-generated test audios:
``
C:\Users\tiger\科大云盘\26-WrightEagle.AI-Speech\test\audio_output
```

**Naming:** `tts_kokoro_s{speaker_id}_{index:02d}.wav`

These files can be used for:
- Future ASR model iteration testing
- RoboCup@Home Task 1 simulation
- Speech interaction system integration verification

## 9. Conclusions

> Auto-generated report. Single-run CPU test results.

### Test Methodology

1. Kokoro TTS (Speaker ID=10) converts 16 test texts to audio
2. Each ASR model processes generated audio
3. Metrics:
   - **Char Accuracy**: SequenceMatcher similarity ratio
   - **Exact Match Rate**: Perfect match proportion
   - **RTF**: Processing time / audio duration (lower = faster)

### Notes

- All tests on **CPU**; GPU would significantly improve speed
- Synthetic speech (TTS-generated) may differ from real human voice
- Different models have different language strengths
- Consider multiple runs for stable averages

---
*Report: 7/7 ASR models tested successfully*
*Generated: 2026-04-11 00:15:58*