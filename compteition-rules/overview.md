# Overview

Pixels to Predictions: DL Vision Challenge is the final course competition for enrolled students, designed to evaluate your ability to build and rigorously assess a deep learning vision system for scientific multiple-choice reasoning. The goal is to develop a SmolVLM-500M-Instruct model (from an official Hugging Face pretrained checkpoint) that uses the provided images and question context to predict the correct answer choice for each test example, with leaderboard performance measured by prediction accuracy.

This competition is governed by strict academic constraints: teams may include up to 2 students, only the provided competition data may be used (no external data), evaluation runs in an offline setting with no internet access, development must remain within Google Colab Free tier and Kaggle Free tier resources, and the maximum number of trainable parameters is capped at 5 million. Participants should start from the provided starter notebook and follow Kaggle guidance here: Getting Started Guide, Competition Documentation, and Submitting Predictions.

# Evaluation

Submissions are scored using classification accuracy on a hidden test set. For each test example, your model must predict a single 0-indexed answer choice.

Accuracy = 1/N \sum_{i=1}^{N} 1[\hat{y}_i = y_i]

where N is the number of test examples, $\hat{y}_i$ is your predicted answer index, and $y_i$ is the ground-truth answer index.

The leaderboard is split into:

- **Public leaderboard**: computed on a public subset of hidden test labels during the competition.
- **Private leaderboard**: computed on a separate hidden subset and used for final ranking.

# Submission File

Your submission must be a CSV file named submission.csv with exactly two columns:

- **id**: test example identifier
- **answer**: predicted 0-indexed integer answer

Use this structure:

id,answer  
test_02333,0  
test_04102,2  
test_00017,1  

# Submission Rules

- One prediction per test id.
- id values must match the provided test file.
- answer must be an integer index valid for that question’s choice set.
- Files with missing columns, extra columns, invalid ids, or non-integer answers may be rejected or scored as invalid.