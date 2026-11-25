# Intent-IQ: Multilingual Intent Classification

A multilingual intent classification system designed to understand user queries in English, Hindi, and Hinglish using a hybrid machine learning approach.

Intent-IQ combines a transformer-based language model with traditional machine learning to improve intent recognition across multilingual and code-mixed text.

## Project Overview

Intent classification is a Natural Language Understanding (NLU) task that identifies the intention behind a user's input.

For example:

| User Query | Predicted Intent |
|---|---|
| Play my favorite song | Play Music |
| Mujhe alarm set karna hai | Set Alarm |
| What is the weather today? | Get Weather |
| Call my mother | Make a Call |

The system is designed to handle multilingual and code-mixed queries, including Romanized Hindi and Hinglish.

## Model Architecture

The final system uses a hybrid ensemble consisting of:

- **MuRIL** — Multilingual Representations for Indian Languages, a transformer model designed to support Indian languages and their variations.
- **TF-IDF** — Word and character-based text feature extraction.
- **Linear SVM** — A Support Vector Machine classifier trained on TF-IDF features.

The predictions from the MuRIL and TF-IDF + SVM models are combined to produce the final intent classification.

### Architecture

```text
              User Query
                  |
          Text Preprocessing
                  |
          +-------+--------+
          |                |
      MuRIL Model      TF-IDF Vectorizer
          |                |
   Transformer Output    Feature Vectors
          |                |
   MuRIL Prediction      Linear SVM
          |                |
          +-------+--------+
                  |
           Ensemble Fusion
                  |
         Final Intent Class
```

## Model Performance

The final ensemble achieved the following result on the local holdout validation dataset:

| Metric | Result |
|---|---:|
| Holdout Macro-F1 Score | **0.8499** |
| Number of Intent Classes | 56 |
| Approximate Model Parameters | **248 Million** |
| Model Type | MuRIL + TF-IDF + SVM Ensemble |

The Macro-F1 score evaluates performance across all intent classes, giving equal importance to each class.

The reported score is from local holdout validation and should not be interpreted as the official competition leaderboard score.

## Dataset

The dataset contains multilingual user queries labelled with their corresponding intent classes.

The training data includes:

- English queries
- Hindi queries
- Romanized Hindi
- Hinglish and code-mixed queries

The dataset is used to train and evaluate the intent classification models.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- MuRIL
- Scikit-learn
- TF-IDF
- Linear Support Vector Machine
- Pandas
- NumPy
- Google Colab

## Project Workflow

1. Load and inspect the multilingual intent dataset.
2. Clean and preprocess the text.
3. Split the dataset into training and validation sets.
4. Train the TF-IDF + SVM classifier.
5. Fine-tune the MuRIL transformer model.
6. Evaluate both models using Macro-F1.
7. Combine the model predictions using ensemble fusion.
8. Evaluate the final ensemble on the holdout validation set.
9. Generate predictions for the competition test dataset.

## Inference

The system accepts a user query and predicts its corresponding intent class.

Example:

```python
query = "play my favorite song"

predicted_intent = predict_intent(query)

print(predicted_intent)
```

The final prediction is generated using the ensemble of MuRIL and TF-IDF + SVM.

## Model Size

The combined model has approximately **248 million parameters**, keeping it within the project's 300-million-parameter limit.

The parameter count refers to the neural model components. TF-IDF vocabulary and SVM support vectors are additional model artifacts and are not counted as neural network parameters.

## Applications

Intent-IQ can be used in:

- Multilingual virtual assistants
- Voice-controlled applications
- Smart device interfaces
- Conversational AI systems
- Natural Language Understanding pipelines
- Intent-based chatbot systems

## Future Improvements

- Improve performance on low-resource intent classes.
- Explore additional multilingual data augmentation.
- Optimize ensemble weights using validation data.
- Improve inference efficiency.
- Deploy the model through a user-friendly web interface.

## Conclusion

Intent-IQ demonstrates how transformer-based language understanding and traditional machine learning can be combined to build a robust multilingual intent classification system.

The final MuRIL + TF-IDF + SVM ensemble achieved a **0.8499 holdout Macro-F1 score** with an approximate model size of **248 million parameters**.

The project highlights the potential of hybrid machine learning architectures for multilingual and code-mixed Natural Language Understanding.
