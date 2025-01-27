# Patent Phrase Matching: A Classification-Based NLP Project

This repository implements a Natural Language Processing (NLP) project that addresses the U.S. Patent Phrase to Phrase Matching problem. The task involves comparing two phrases and determining their semantic similarity based on patent classifications. This project demonstrates how to solve the problem by treating it as a classification task, leveraging powerful tools like pandas, transformers, and numpy.

## Problem Description

The objective is to assess the semantic similarity between two phrases (e.g., TEXT1 and TEXT2) and assign a similarity score. A score of:

- 1.0 indicates identical meaning.
- 0.5 indicates somewhat similar meaning.
- 0.0 indicates completely different meaning.

Instead of directly predicting a continuous similarity score, we reframe the problem as a classification task. The question is reformulated as:

For the following text...: "TEXT1: abatement; TEXT2: eliminating process"
...choose a category of meaning similarity: Different, Similar, or Identical.


## Project Workflow

- ### Data Preparation
    - Load the dataset (e.g., a .csv file) using pandas. Clean and preprocess the text data to create a single combined sentence for each phrase pair:
     - Combined Sentence: TEXT1: <phrase1>; TEXT2: <phrase2>

    - Convert similarity scores (0.0, 0.5, 1.0) into categories: Different, Similar, Identical.

- ### Model Fine-Tuning
    - Fine-tune the pre-trained transformer model, microsoft/deberta-v3-small, for classification using Hugging Face's transformers library.

- ### Evaluation
    - Evaluate the model using the Pearson correlation coefficient (r) between the predicted and actual similarity scores.

- ### Predictions
     - Use the trained model to classify new phrase pairs and predict their similarity categories.

![Data preprocessing (2)](https://github.com/user-attachments/assets/fa2e2727-8490-4d77-ba9b-afbc0ac7f27d)


 
## Evaluation Metric

The Pearson correlation coefficient (r) measures the degree of relationship between two variables. It ranges between:

 - +1: Perfect positive correlation (predictions match actual scores exactly).
 - 0: No correlation.
 - -1: Perfect inverse correlation (predictions are opposite to actual scores).
It is the most widely used measure for evaluating the relationship between predicted and actual similarity scores.

The formula for Pearson correlation is:

𝑟 = Cov(𝑋, 𝑌)/𝜎𝑋𝜎𝑌


## Skills used

 - pandas: For data loading, cleaning, and preprocessing.
 - transformers: For using Hugging Face's AutoTokenizer and Trainer APIs to tokenize data and fine-tune the DeBERTa model.
 - numpy: For numerical computations, especially during evaluation.


## Results
The fine-tuned DeBERTa model successfully classifies phrase pairs into Different, Similar, or Identical categories. The evaluation metric, Pearson correlation coefficient (r), demonstrates the alignment between predicted and actual similarity scores.
