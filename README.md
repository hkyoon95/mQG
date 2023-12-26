# muti-Question Generation model

## Overview
This repo is the official implementation of [Diversity Enhanced Narrative Question Generation for Storybooks](https://aclanthology.org/2023.emnlp-main.31.pdf).

## Tested Environment
- Python 3.10.6
- Pytorch-lightning 1.7.7

## Instructions

First, install the Python dependencies:

    pip install -r requirements.txt

Second, Download the checkpoint from the following Google Drive link: 

mQG

https://drive.google.com/file/d/1-GQzRTfiUASz-de6Ikvz10xnNNO1MOYY/view?usp=share_link

Answerability Evaluation Model

https://drive.google.com/file/d/1-HCf8gJZ7V3uPgb3dqDDNcmnOi0MeS7U/view?usp=share_link

Third, run mQG_generate.py

If you'd like to train the model, you can download the FairytaleQA dataset from [here](https://huggingface.co/datasets/WorkInTheDark/FairytaleQA).

## Citation

```
@inproceedings{yoon-bak-2023-diversity,
    title = "Diversity Enhanced Narrative Question Generation for Storybooks",
    author = "Yoon, Hokeun  and Bak, JinYeong",
    year = "2023",
    publisher = "Association for Computational Linguistics",
}
```
