# Multimodal Science Question-Answering Competition

This competition uses a multimodal science question-answering dataset in which each example combines an image with textual context and multiple-choice options. Participants must predict the correct answer choice index for each question. The task requires both visual interpretation (e.g., diagrams, maps, charts) and language reasoning (question, hint, and lecture/context text).

## Data Splits

The data is organized into three splits:

- **Train**: labeled examples for model fitting.  
- **Validation**: labeled examples for model selection and tuning.  
- **Test**: unlabeled examples used for leaderboard scoring.  

## Dataset Schema

Each row corresponds to one question instance and includes:

- **id**: unique example identifier.  
- **image_path**: relative path to the associated image file.  
- **question**: question text.  
- **choices**: JSON list of candidate answers.  
- **num_choices**: number of options in choices.  
- **answer**: 0-indexed correct option (available only in train/validation).  
- **hint, lecture**: optional textual context fields.  
- **task, grade, subject, topic, category, skill**: pedagogical metadata.  

## Image Storage

Images are stored by split under:

data/images/{train,val,test}

## Submission Format

A sample_submission.csv is provided with the required submission schema:

id  
answer (0-indexed predicted choice)

## Evaluation

The hidden test labels are withheld for evaluation, with leaderboard scoring performed on hidden ground truth.