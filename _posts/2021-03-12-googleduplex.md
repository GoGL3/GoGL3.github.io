---
title: "[NLP] Google Duplex - 'Jarvis' comes true?"
date: 2021-03-12 10:000 -0400
author : 정여진
categories :
  - paper-review
  - NLP
tags :
  - NLP
  - deep-learning
---

![2021-03-12](/assets/2021-03-12-duplex1.png)

You wake up and it's 8:00 a.m. You have to hurry to get to work on time. You get ready, grab some sandwiches, and rush to your car. While driving, you realize that you have a date tonight and need to make a reservation at some fancy restaurant. But remember, you are on a rush and you have no time to make a call. Google Assistant can help you. You can say, "Hey Google, make a reservation for two at 6:00p.m. at Sushi-in." Then Boom! Google Assistant **calls** the restaturant, **generates** human voice and makes a reservation.

Google Duplex is a name for the technology supporting Google Assistant. This service was first introduced in 2018 and has been mainly used in _booking_ with human-like phone calls. 


## Technology

![2021-03-12](/assets/2021-03-12-duplex2.png)

Of course, some insanely high quality NLP is used in Duplex. First, the device has to understand what the person is saying even with some background noise (speech-to-text). Next, the context of the complex sentences needs to be analyzed and correct responses need to be  generated (natural language understanding). Finally, this response has to be transformed into human-like voice (text-to-speech). Google Duplex trained the models **separately** for each task (one for business appointment, one for reservation etc). Models used are recurrent neural network (RNN) combined with Google’s Automatic Speech Recognition (ASR) technology, parameters of the conversation (e.g., desired time, names), and a text-to-speech (TTS) system.

### Further reading:
- Google AI [blog](https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html)
- Google's natural language understanding [post](https://blog.google/products/search/search-language-understanding-bert/)

### References
- [NLP in the Real World: Google Duplex](https://towardsdatascience.com/nlp-in-the-real-world-google-duplex-d96160d3770b)

