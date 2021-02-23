---
title: "[NLP] Difficulties in applying NLP models to Korean data"
date: 2021-02-24 06:000 -0400
author : 조경민
categories :
  - NLP
  - Korean
  - Tokenization
  - Stopwords

---

## Difficulties of analyzing KOREAN text data



#### 1. Tokenization

Although most of the NLP models already offered a pre-trained model for multilingual data, it is still difficult to put it directly into Korean. Korean is a complex language, so there are many aspects that the Tokenizer used here does not fit well. No matter how well pre-training has done, the performance will be terrible if you don't make subword vocab well.

There are several widely used subword tokenizers - BPE(Byte-Pair Encoding Tokenization), SentencePiece, WordPiece, Huggingface.... etc. These methods are designed to solve out-of-vocabularies (OOV) problems, but are not yet perfect. Emojis or Chinese characters can cause OOV problems when analyzing text data. How to handle those has a big impact on model performance. In order to use NLP models, it is necessary to directly establish subword tokenization of Korean data. However, there are only a small number of packages that directly performs these kinds of tokenization dealing with KOREAN texts. To make matters worse, some existing packages often have problems, such as lack of sophistication or complicated processes to proceed the analysis. Therefore, when dealing with Korean data, the process of tokenization itself is not smooth. :(

Here are some github repos that offer tool for analyzing Korean text data. - KoBERT, KoELECTRA, Mecab tokenizer.... etc.

<https://github.com/BM-K/KoSentenceBERT_SKTBERT>

<https://github.com/SKTBrain/KoBERT>

<https://github.com/monologg/KoELECTRA>

<https://github.com/Gyunstorys/nlp-api>

<https://github.com/domyounglee/korbert-mecab-multigpu>

<https://monologg.kr/2020/04/27/wordpiece-vocab/>





#### 2. Stopwords / Morphological analysis

Unfortunately, Korean does not have an elaborate dictionary of stopwords like English. Although some analysts already made a dictionary of frequently used stopwords, you need to individually add the stopwords appeared in data, if you want to analyze a large corpus. It is very annoying and time-consuming to update the stopwords dictionary whenever data changes. Therefore, we need to build a more generalized and robust dictionary of Korean stopwords. 

Also, there are many types of morphology analyzer in Korean, so users must directly select an analyzer that fits their data. For example, a Python package called 'konlpy' also has several morphemes, such as Twitter, Komoran, Kkma, and Hannanum. Because each analyzer has different morphemes, and the time required for the analysis varies widely, analysts should take time to think about what should be used to suit the purpose of the analysis. 

As such, Korean has yet to be settled on the dictionary for stopwords or the analysis of morphemes, so there are many things that need to be improved when performing NLP.

<https://datascienceschool.net/03%20machine%20learning/03.01.02%20KoNLPy%20%ED%95%9C%EA%B5%AD%EC%96%B4%20%EC%B2%98%EB%A6%AC%20%ED%8C%A8%ED%82%A4%EC%A7%80.html>

<https://bab2min.tistory.com/544>





#### 3. Time restriction

When an already released package does not fit well with your data and if you try to code again from scratch, you have to build a vocab.txt for training and pre-train the model to use it for fine-tuning. But this is a very time-consuming task. It may be too difficult to even try this work with the laptops that students generally have. And even if existing packages are utilized as they are, analysis can be difficult because the time it takes can be very long depending on the size of the data or the performance of the package function. 





#### Conclusion

There are still many unresolved difficulties in applying NLP to Korean data. Therefore, it will be necessary to make good use of existing models and new SOTA methodologies to build generalized packages that can be applied to most of Korean data.
