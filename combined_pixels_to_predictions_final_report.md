# Pixels to Predictions: SmolVLM Candidate Yes/No for Visual Science QA

**Author:** Preyansh Agrawal  
**Affiliation:** New York University  
**Competition:** DL Spring 2026 — Pixels to Predictions  
**Base model:** `HuggingFaceTB/SmolVLM-500M-Instruct`  
**Constraint:** at most 5 million trainable parameters  
**Final notebook:** `DL Final Notebook.ipynb`  
**Best public leaderboard accuracy:** **76.6659%**

## Abstract

This project studies parameter-efficient fine-tuning of `HuggingFaceTB/SmolVLM-500M-Instruct` for scientific multimodal multiple-choice reasoning. Each example contains an image, a science question, and 2 to 5 answer choices. The model must predict the correct 0-indexed answer choice under strict competition constraints: no external data, no external models, offline evaluation, and at most 5 million trainable parameters.

The system evolved from a direct answer-letter formulation into a candidate-level yes/no verification formulation. The direct formulation asked the model to output `A/B/C/D`, but validation collapsed almost entirely to answer index 0 and achieved only 33.40% accuracy. The final formulation expanded each multiple-choice question into one binary candidate-verification example per answer choice. During inference, each candidate was scored independently using the model's `log P(" Yes")`, and the answer choice with the highest yes score was selected.

The final notebook achieved **76.6659% public leaderboard accuracy**. The final run used candidate yes/no training, answer-only supervision, balanced answer-index sampling, rank-16 LoRA on attention projections plus `out_proj`, no metadata, no lecture or hint context, manual 384×384 image resizing, `image_seq_len=64`, bf16 training, tf32 matmul, and a final train+validation training run.

## 1. Introduction

The Pixels to Predictions challenge requires a model to answer scientific multiple-choice questions using both an image and textual question context. Many examples require visual interpretation of diagrams, charts, maps, or scientific illustrations, while the number of answer choices varies by question.

The competition restricts the modeling space substantially. Participants must use the official `SmolVLM-500M-Instruct` checkpoint, cannot use external data or auxiliary models, and must keep trainable parameters below 5 million. These constraints make full fine-tuning impossible and place more emphasis on prompt design, parameter-efficient adaptation, robust sampling, and careful evaluation.

The project’s main modeling lesson was that the obvious direct answer-index formulation was unstable. Asking the model to choose `A/B/C/D` directly caused the model to prefer the first answer position almost all the time. The stronger formulation treated each answer choice as a candidate and trained the model to decide whether that candidate was correct. This candidate yes/no setup became the core of the final system.

## 2. Dataset and Task Format

The competition provides train, validation, and test splits. Each row contains `id`, `image_path`, `question`, `choices`, `num_choices`, and, for train and validation, a gold `answer` index. Additional optional fields include `hint`, `lecture`, `solution`, `task`, `grade`, `subject`, `topic`, `category`, and `skill`.

The final submission is a CSV named `submission.csv` with exactly two columns:

```csv
id,answer
test_02333,0
test_04102,2
test_00017,1
```

The primary metric is classification accuracy on hidden test labels.

Two properties affected the final design. First, answer indices were skewed, with indices 0 and 1 much more common than index 4. Early models exploited this skew and collapsed toward answer index 0. Second, questions had a variable number of choices, from 2 to 5, which made the candidate yes/no formulation convenient because each choice could be scored independently.

## 3. Model

The base model is `HuggingFaceTB/SmolVLM-500M-Instruct`, a compact vision-language model with an image encoder, vision-language projection components, and a causal language decoder. Its small size makes it suitable for Colab and Kaggle free-tier constraints, but also means prompt noise, weak visual perception, and answer-position bias can significantly affect performance.

The final system used LoRA rather than full fine-tuning. The final LoRA configuration was:

```python
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_DORA = False
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
```

This produced approximately 4,456,448 trainable parameters, under the 5 million parameter cap. The `q_proj`, `k_proj`, `v_proj`, and `o_proj` modules adapt attention behavior, while adding `out_proj` allowed the adapter to affect visual or attention output projection behavior within the parameter budget.

## 4. Final Method

### 4.1 Candidate yes/no formulation

Each question with `K` answer choices was expanded into `K` binary examples. The final prompt template was:

```text
<image>

Question:
{question text}

Candidate answer:
{candidate text}

Is the candidate answer correct? Reply with Yes or No only.
Answer:
```

The target was ` Yes` for the correct candidate and ` No` for incorrect candidates. At inference time, each candidate was scored with the model’s next-token `log P(" Yes")`; the candidate with the highest yes score became the final predicted answer index.

### 4.2 Answer-only supervision

The collator encoded both prompt-only text and full prompt-plus-target text. It masked all prompt and padding tokens in the label tensor with `-100`, so the loss was computed only on the target token. This made the model learn the binary decision rather than reconstructing the input prompt.

### 4.3 Balanced answer-index sampling

The final system used inverse-frequency sampling over the original answer index. A `WeightedRandomSampler` sampled candidate rows with weights based on the gold answer position of the original question. This reduced answer-position bias and prevented the model from collapsing toward answer index 0.

### 4.4 Final configuration

```python
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
TRAIN_ON_VAL_TOO = True

IMAGE_LONGEST_EDGE = 384
IMAGE_SEQ_LEN = 64
MAX_LECTURE_CHARS = 0
MAX_HINT_CHARS = 0
USE_METADATA = False

EPOCHS = 1
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER = "linear"
MAX_GRAD_NORM = 1.0

USE_BF16 = True
USE_FP16 = False
USE_4BIT = False
TF32 = True
GRADIENT_CHECKPOINTING = False
DATALOADER_WORKERS = 4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_DORA = False
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
```

Images were manually resized to 384×384 with bicubic interpolation before processing. The final processor used `image_seq_len=64`. Earlier experiments with `image_seq_len=36`, processor overrides, and different image sizes were less successful.

### 4.5 Evaluation and submission

Evaluation was batched. For each batch of original questions, the evaluator built all candidate prompts, scored each candidate with `log P(" Yes")`, regrouped scores by original question, and selected the argmax candidate index.

The final test prediction distribution was:

| Predicted answer index | Share |
|---:|---:|
| 0 | 0.359127 |
| 1 | 0.356151 |
| 2 | 0.204365 |
| 3 | 0.072421 |
| 4 | 0.007937 |

This distribution was not collapsed and closely matched the expected answer-index pattern.

## 5. Experimentation

### 5.1 Direct answer-letter baseline

The first baseline prompted the model to output one answer letter:

```text
Question: ...
Choices:
A. ...
B. ...
C. ...
Answer:
```

At inference time, the model scored next-token probabilities for answer-letter tokens. This failed because the model strongly preferred the first answer token.

```text
Validation accuracy: 0.3340
Prediction distribution:
0    0.9933
3    0.0067

Gold distribution:
0    0.3311
1    0.3550
2    0.2357
3    0.0716
4    0.0067
```

This result was essentially equivalent to always predicting answer index 0.

### 5.2 Candidate yes/no initial runs

Switching to candidate yes/no fixed the collapse. Early prediction distributions became much more plausible, but initial validation accuracy was only about 44.18%, showing that the formulation was better but the training recipe still needed improvement.

### 5.3 Batch size and gradient accumulation

A major intermediate improvement came from reducing the effective batch size. Large-batch training generalized poorly.

| Setting | Effective batch | Validation accuracy |
|---|---:|---:|
| `TRAIN_BATCH_SIZE=8`, `GRAD_ACCUM=4` | 32 | 54.4% |
| `TRAIN_BATCH_SIZE=8`, `GRAD_ACCUM=2` | 16 | 60.11% |

The batch 8, accumulation 2 run had a healthy loss curve, dropping from 5.347117 at step 20 to 0.098520 at step 300.

### 5.4 Metadata experiment

Metadata fields such as `subject`, `grade`, `topic`, `category`, and `skill` were tested as extra context. They did not improve performance. The best configurations used `USE_METADATA=False`.

### 5.5 Lecture and hint context

Longer context was tested with `MAX_LECTURE_CHARS=900` and `MAX_HINT_CHARS=250`. This produced essentially the same validation accuracy as the shorter setting. The final best notebook removed both lecture and hint context entirely with `MAX_LECTURE_CHARS=0` and `MAX_HINT_CHARS=0`.

### 5.6 MLP LoRA targets

Expanding LoRA to MLP layers was tested with rank 8:

```python
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
```

This reduced validation accuracy to about 55%, below the attention-only baseline. The experiment was rejected.

### 5.7 DoRA

DoRA was tested on the best attention-only LoRA setup with `USE_DORA=True`. It achieved about 58% validation accuracy, below the standard LoRA baseline. The final model used `USE_DORA=False`.

### 5.8 Larger image size

Larger image preprocessing was tested with `IMAGE_LONGEST_EDGE=512` and `IMAGE_SEQ_LEN=64`. This reduced validation accuracy to about 50%. The final model retained 384×384 images.

### 5.9 Prompt wording

Several candidate verification phrasings were explored. The final wording was:

```text
Is the candidate answer correct? Reply with Yes or No only.
```

This wording was used consistently during training and inference.

### 5.10 Pipeline-alignment improvements

A later comparison with a stronger config-driven pipeline revealed several useful differences. The final notebook aligned with those findings by using learning rate `5e-5`, warmup ratio `0.05`, a linear schedule, no 4-bit quantization for the final run, no metadata, no hint, no lecture, no full list of choices in the prompt, `image_seq_len=64`, `out_proj` in LoRA targets, and manual image resizing to 384×384.

### 5.11 Caption augmentation experiment

A self-captioning experiment generated one-sentence image descriptions using the base SmolVLM model and injected those captions into the prompt. This hurt performance. The caption-based run achieved about 66%, much worse than the captionless final method. The likely reason is that the same small model produced noisy or hallucinated captions, and the model then learned from its own unreliable descriptions.

### 5.12 Final train+validation run

After validation experiments identified the best setup, the final notebook trained on the combined train and validation splits with `TRAIN_ON_VAL_TOO=True`. This provided more labeled data for the final hidden-test submission. The final run used 1 epoch, batch size 4, gradient accumulation 1, learning rate `5e-5`, bf16, tf32, attention LoRA plus `out_proj`, and no extra context fields.

The result was **76.6659% public leaderboard accuracy**.

## 6. Results

| Run / experiment | Main change | Result | Outcome |
|---|---|---:|---|
| Direct letter scoring | Predict `A/B/C/D` directly | 33.40% validation | Failed due to label-0 collapse |
| Candidate yes/no initial | Score each candidate independently | 44.18% validation | Fixed collapse but weak checkpoint |
| Improved yes/no training | Better batch setup | 54.4% validation | Clear improvement |
| Batch 8, grad accum 2 | Effective batch 16 | 60.11% validation | Best intermediate validation setup |
| Metadata on | Add subject/grade/topic metadata | Worse than metadata off | Rejected |
| Longer lecture/hint | 900 lecture chars, 250 hint chars | No improvement | Rejected |
| MLP LoRA | Rank 8 with MLP targets | 55% validation | Rejected |
| DoRA | Standard LoRA replaced by DoRA | 58% validation | Rejected |
| 512 image | 512 longest edge | 50% validation | Rejected |
| Caption augmentation | Self-generated image captions | 66% leaderboard | Rejected |
| Refined pipeline-aligned notebook | Prompt/preprocessing alignment | 75% leaderboard | Strong intermediate run |
| Final train+val notebook | Best yes/no pipeline, no context, LoRA + `out_proj` | **76.6659% leaderboard** | Final best run |

## 7. Discussion

The main finding is that task formulation mattered more than increasing model complexity. Direct answer-letter prediction was natural but unstable. Candidate yes/no transformed the task into binary verification and removed the direct dependence on answer-index tokens.

The second finding is that prompt minimalism worked better than adding context. Metadata, hint, lecture, and generated captions all failed to improve the best configuration. For this small VLM, additional text often acted as noise rather than useful context.

The third finding is that adaptation choices need to be validated empirically. MLP LoRA, DoRA, and 512-pixel images were plausible ideas, but all underperformed the simpler rank-16 attention-focused LoRA setup with `out_proj`.

The fourth finding is that train/eval pipeline consistency matters. The final gains depended on consistent prompt formatting, consistent manual image preprocessing, correct image-token configuration, and using the same candidate yes/no scoring path at training and inference time.

## 8. Limitations

The final 76.6659% result is a public leaderboard score. Private leaderboard performance may differ. Since the final model trained on train+validation, there is no held-out validation score for the exact final checkpoint.

The experiment search was not exhaustive. Other LoRA target combinations, score calibration methods, image augmentations, choice-order permutations, or ensembling strategies may improve performance further.

The captioning experiment used only the allowed base SmolVLM model to generate captions. A stronger external captioner might help, but would violate the competition’s allowed-model constraints.

The final run was executed on an A100-class GPU with bf16 and tf32 enabled. Exact speed and small numerical differences may vary on other hardware.

## 9. Conclusion

This project developed a competitive SmolVLM-based solution for visual scientific multiple-choice reasoning under strict academic and compute constraints. The initial direct answer-letter baseline collapsed almost entirely to answer index 0. The key improvement was a candidate yes/no formulation with balanced answer-index sampling.

The final system used answer-only supervision, rank-16 LoRA, attention targets plus `out_proj`, 384×384 images, no metadata, no lecture or hint context, bf16 training, and a final train+validation training run. It achieved **76.6659% public leaderboard accuracy**.

The main lesson is that for small VLMs on multiple-choice reasoning, stable task reformulation and evaluation-aware scoring can matter more than adding context, larger images, or more complex adapters.

## 10. Reproducibility Notes

The final notebook should preserve the following settings:

```python
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
TRAIN_ON_VAL_TOO = True
USE_METADATA = False
MAX_LECTURE_CHARS = 0
MAX_HINT_CHARS = 0
IMAGE_LONGEST_EDGE = 384
IMAGE_SEQ_LEN = 64
EPOCHS = 1
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER = "linear"
MAX_GRAD_NORM = 1.0
USE_BF16 = True
USE_FP16 = False
USE_4BIT = False
TF32 = True
GRADIENT_CHECKPOINTING = False
DATALOADER_WORKERS = 4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_DORA = False
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
SCORING = "last-token log P(' Yes')"
```

Final artifacts:

```python
final_adapter_dir = OUTPUT_DIR / "final_adapter"
trainer.save_model(str(final_adapter_dir))
processor.save_pretrained(str(final_adapter_dir))

submission_path = OUTPUT_DIR / "submission.csv"
submission_df.to_csv(submission_path, index=False)
```

Submission sanity checks should verify that columns are exactly `id` and `answer`, row count matches the sample submission, test IDs match sample submission IDs, every answer is an integer, and every answer is within the valid range for that question’s choice set.

## Acknowledgments

This report draft is based on the final notebook, validation outputs, leaderboard result, and experiment notes collected during development. The report structure follows a standard research-report format with abstract, introduction, dataset, model, experimentation, results, limitations, conclusion, and reproducibility sections.
