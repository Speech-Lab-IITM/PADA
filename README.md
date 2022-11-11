# [PADA: Pruning Assisted Domain Adaptation](https://arxiv.org/abs/2203.16965)

Paper tile: PADA: Pruning Assisted Domain Adaptation for Self-Supervised Speech Representations. To Appear at IEEE SLT 2022. [arxiv link](https://arxiv.org/abs/2203.16965).

As a part of this work, we study the effects of different pruning strategies (TAG, TAW and CD-TAW) in the PADA framework. Also, we study the effects of 3 variants of pruning frequencies (Once, Iterative, Dynamic Iterative) which can be applied along-side each of these pruning strategies.

Some of the major contributions of our work include:

* Cross-Domain Task-Aware pruning **(CD-TAW)** where publicly available models fine-tuned on the high-resource datasets, available in the public domain are used to derive highly task relevant masks which in-turn help solve the problem of domain adaptation using the PADA framework. While we know of several works which emphasize the use of pre-trained models, as a part of this work, we also push for the use of publicly available fine-tuned that help solve task-specific problems.
* Dynamic iterative pruning approach: as the number of model updates increase, we reduce the pruning rate. This is because, as the model parameters are adjusted to the target domain, zeroing out the same number of parameters continously would force the model to focus on learning the same set of parameters. We therefore, monotonically decay the pruning rate to facilitate better adaptation of the model to the target domain. This approach in most cases performs better than other pruning frequencies.
* We analyze the performance of different pruning strategies and frequencies for domain adaptation on pre-trained speech SSL models. We base our experiments with the practical assumption that only limited amounts of target-domain labeled data is available and no other large-scale unlabelled corpus is available from the target domain.

## Data Preparation

The low-resource datasets we used as a part of this work are:

* Switchboard data: We'd performed our experiments on 2-hour and 30-hour subsets of the Switchboard dataset. We clean the text of the Switchboard data by removing filler-words and punctuations. Also, we convert all the text to uppercase. To perform this, we use a custom designed script. Please run the `clean_swbd_text.sh` file by passing the text file path to it. This script also generates the `dict.ltr.txt` which is the vocabulary dictionary needed in the finetuning task. The wav2vec2.0 data preparation was done using the standard procedure mentioned in [fairseq](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec). Evaluation was done on the test data of the Switchboard dataset, whose text was cleaned using the same script.
* Hindi data: We use a 7-hour subset of the 50-hour hindi data released as a part of the [Hindi-ASR-Challenge](https://sites.google.com/view/asr-challenge). The wav2vec2.0 data preparation was done using the standard procedure mentioned in [fairseq](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec).

## Running PADA

### Pre-requisites

Install the fairseq toolkit, as all of our experiments are based run on models trained on fairseq. Instructions for installation of the same can be found [here](https://github.com/pytorch/fairseq).

### Extracting the mask

While there is no pruning mask necessary for Task-Agnostic pruning (TAG), pruning masks are necessary for Task-Aware pruning (TAW) and Cross-Domain Task-Aware pruning (CD-TAW). To extract the mask, we run the evaluation script, that has been modified to facilitate the same. The file `infer.py` needs to be replace the existing `infer.py` file in the `examples/speech_recognition` folder of your fairseq installation.

The lines `331-338` of `infer.py`  are the ones which aid in extracting the state_dict carrying the mask. Please edit the parameters in these lines according to the experiment you wish to run.

To run the `infer.py` one needs to pass the fine-tuned model from which the initial pruning mask needs to be extracted. In case of TAW, one needs to run the ASR finetuning of the pre-trained model on the target-domain data itself. After one full round of finetuning, we extract the mask from the fine-tuned model, apply the mask on the pre-trained model, zero out the weights specified in the pre-trained model as specified by the mask and then run the fine-tuning stage using PADA on the pre-trained model with certain weights zeroed out.

Clearly, TAW is compute intensive as it involves 2 rounds of fine-tuning on the target domain. CD-TAW alleviates this issue by picking the initial pruning mask from a model fine-tuned on out-of-domain data, which is publicly available. In most cases, CD-TAW outperforms TAW as the mask has been taken from a model fine-tuned on fairly large amouts of data compared to the small amount of data available in the target-domain. This is because, in our analysis we realize that when the target-domain and the high-resource-domain are from the same language, taking the mask from CD-TAW helps, as the model is more 'task-aware' (as it has seen more data in the finetuning task).

The fine-tuned models (for CD-TAW) we used in order to extract masks were made available by [fairseq](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec). Following are the fine-tuned models that we've used:

1. wav2vec-2.0 LV-60k on LS-100h : This is a wav2vec-2.0 LARGE model, that has been pre-trained on Libri-Light data and fine-tuned on the 100 hour split of the LibriSpeech dataset. Model can be downloaded from [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_100h_new.pt).
2. wav2vec-2.0 LV-60k on LS-960h : This is a wav2vec-2.0 LARGE model, that has been pre-trained on Libri-Light data and fine-tuned on the entire 960 hours of the LibriSpeech dataset. Model can be downloaded from [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_vox_960h_new.pt).
3. wav2vec-2.0 LS-960 on LS-100h : This is a wav2vec-2.0 BASE model, that has been pre-trained on the entire 960 hours of the LibriSpeech dataset and fine-tuned on the 100 hour split of the LibriSpeech dataset. Model can be downloaded from [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt).
4. XLSR-53 on CommonVoice (CV) and Babel: This is a wav2vec-2.0 LARGE model, that has been pre-trained on Multi-lingual LibriSpeech (MLS), CV and Babel and fine-tuned on a subset of CV and Babel as mentioned in [this work](https://arxiv.org/abs/2109.11680). Model can be downloaded from [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/zero_shot/phonetisaurus_40lang_m10.pt).

The state_dict containing the mask would be saved upon successful execution of the following command:

```sh
python examples/speech_recognition/infer.py /path/to/test/data --task audio_finetuning \
--nbest 1 --path /path/to/the/finetuned/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter
```

Once the state_dict has been saved having the mask extracted, what is left would be applying the mask on the pre-trained model and fine-tuning on the target-domain data.

### Applying the mask on the pre-trained model and finetuning

Replace the `train.py` in the `fairseq-cli` folder in your fairseq installation by the one provided in this repository.

Modify the lines `253-268` in `train.py` according to the pruning strategy of your choice (TAG / TAW / CD-TAW).

Similarly, modify the lines `415-416` and `509-530` according to the pruning frequency (Once / Iterative / Dynamic Iteractive) you wish to use.

To start the CTC based fine-tuning on the target-domain using PADA, execute the following command:

```sh
fairseq-hydra-train \
    task.data=/path/to/the/finetuning/data \
    model.w2v_path=/path/to/the/pre-trained/model.pt \
    --config-dir /path/to/the/config/directory/in/this/repository \
    --config-name config_name
```

Change the `config_name` to the name of the yaml file you're using based on the dataset you're dealing with and the pre-trained model at hand. The files in the `config` directory have been named accordingly.

Note: you can simulate 24 GPUs by using k GPUs and adding command line parameters (before `--config-dir`) `distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 24/k.

## Evaluating a CTC model

Data preparation process for the test data is same as the data preparation process followed in the finetuning stage.

To decode without any language model, proceed with the execution of the following command:

```sh
python examples/speech_recognition/infer.py /path/to/test/data --task audio_finetuning \
--nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter
```
