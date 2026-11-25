# Intent-IQ 🤖
### Multilingual Intent Classification for Hinglish and Indian Languages

Intent-IQ is a machine learning project designed to identify the intent behind user queries written in English, Hindi, and Hinglish.

The project uses natural language processing and machine learning techniques to classify text into **56 different intent categories**. It aims to provide fast and efficient intent recognition for conversational AI applications.

---

## 🚀 Project Overview

Understanding user intent is an important part of building intelligent chatbots and conversational systems.

Intent-IQ takes a user query as input and predicts the most likely intent category.

For example:

| User Query | Predicted Intent |
|---|---|
| "What's the weather today?" | Weather-related intent |
| "Mujhe music sunna hai" | Music-related intent |
| "Set an alarm for 7 AM" | Alarm-related intent |

The examples above illustrate the task; exact intent labels depend on the dataset.

---

## 🧠 Model Architecture

The final inference pipeline uses:

**Word + Character TF-IDF → Linear Support Vector Machine (SVM)**

### Pipeline

1. Text preprocessing and normalization
2. Word-level TF-IDF feature extraction
3. Character-level TF-IDF feature extraction
4. Linear SVM classification
5. Intent prediction

### Why this approach?

- Lightweight and efficient
- Fast inference
- Suitable for multilingual and Romanized text
- Character n-grams can help handle spelling variations
- Does not# theash
