# Wav2Vec2-ASR-Amharic-CV-16.1
This the code for fine tuning Wav2Vec2-Large-XLS-R-300M for building an ASR for Amharic Language using Mozilla Common Voice 16.1 Dataset. Implementation of the model to a web application using Gradio Library.
---
license: apache-2.0
Base Model: facebook/wav2vec2-xls-r-300m
Datasets: common_voice_16_1
---
This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the common_voice_16_1 dataset.
It achieves the following results on the evaluation set:
- Validation Loss : 1.6333
- Word Error Rate (WER): 0.8639

## Model description

# Wav2Vec2-XLS-R-300M

[Facebook's Wav2Vec2 XLS-R](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) counting **300 million** parameters.

![model image](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/xls_r.png)

XLS-R is Facebook AI's large-scale multilingual pretrained model for speech (the "XLM-R for Speech"). It is pretrained on 436k hours of unlabeled speech, including VoxPopuli, MLS, CommonVoice, BABEL, and VoxLingua107. It uses the wav2vec 2.0 objective, in 128 languages. When using the model make sure that your speech input is sampled at 16kHz. 

**Note**: This model should be fine-tuned on a downstream task, like Automatic Speech Recognition, Translation, or Classification. Check out [**this blog**](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) for more information about ASR.

[XLS-R Paper](https://arxiv.org/abs/2111.09296)

Authors: Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli


## Intended uses & limitations

This model is a general purpose speech to text conversion model. There can be several use cases of this model such as in eduation, business, healthcare, government sectors etc.
The model can be specifically designed for this sectors.

## Training and evaluation data

Datasets: common_voice_16_1
Language Used: Amharic
Total Recorded Hours: 3
Total Validated Hours: 2

It had two splits of Train and Test data.
Training Data: 638 Sentences (5-10 ms)
Testing Data: 162 Sentences (5-10 ms)
Sampling Rate: 16000 Hz

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

### Results on Test Data

The average Word Error Rate (WER) on the Test Data is: 0.7805203468979319
---
TEST WER: 78.05%
---
### Framework versions

- Transformers 4.42.0
- Pytorch 2.3.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
- Gradio 4.44.0

## Implementing the model into Web Application

To implement the model, I have used Gradio model to host and deploy the model into a web application.
I have created the interface of the Web Application through Gradio. Rest the API integration is handeled by the library itself.

