# Wav2Vec2-ASR-Amharic-CV-16.1
This the code for fine tuning Wav2Vec2-Large-XLS-R-300M for building an ASR for Amharic Language using Mozilla Common Voice 16.1 Dataset. Implementation of the model to a web application using Gradio Library.
---
license: apache-2.0
base_model: facebook/wav2vec2-xls-r-300m
datasets:
- common_voice_16_1
metrics:
- wer
model-index:
- name: wav2vec2-large-xls-r-300m-amharic-demo-colab
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: common_voice_16_1
      type: common_voice_16_1
      config: am
      split: test
      args: am
    metrics:
    - name: Wer
      type: wer
      value: 0.8639092728485657
---

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the common_voice_16_1 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6333
- Wer: 0.8639

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 100
- num_epochs: 60
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 12.6948       | 5.0   | 100  | 4.1621          | 1.0    |
| 4.1026        | 10.0  | 200  | 4.0365          | 1.0    |
| 4.0037        | 15.0  | 300  | 3.9726          | 1.0007 |
| 3.9485        | 20.0  | 400  | 3.9524          | 1.0007 |
| 3.4635        | 25.0  | 500  | 2.4384          | 0.9980 |
| 1.1709        | 30.0  | 600  | 1.6987          | 0.9453 |
| 0.4955        | 35.0  | 700  | 1.5927          | 0.9073 |
| 0.3163        | 40.0  | 800  | 1.6750          | 0.8833 |
| 0.2372        | 45.0  | 900  | 1.6683          | 0.8813 |
| 0.1896        | 50.0  | 1000 | 1.6555          | 0.8779 |
| 0.1619        | 55.0  | 1100 | 1.6312          | 0.8819 |
| 0.1473        | 60.0  | 1200 | 1.6333          | 0.8639 |


### Framework versions

- Transformers 4.42.0
- Pytorch 2.3.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
