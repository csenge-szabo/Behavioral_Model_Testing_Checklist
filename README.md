# Behavioral Testing of SRL Models using Checklist

## Author: Csenge Szabó

## Introduction
This project implements Checklist methods to test Semantic Role Labeling (SRL) models developed in the Advanced NLP Master's course at VU Amsterdam. It aims to assess transformer-based models' capabilities in SRL.

For the detailed project report and results, see `Challenging_SRL_Models_Csenge_Szabo.pdf`.

## Data

For this project the fine-tuned DistilBert models were trained using the [Universal Proposition Banks version 1.0](https://universalpropositions.github.io) for English language, which was created with the aim to study Semantic Role Labelling (SRL). The aim of this work is to explore and evaluate the models' performance on SRL using CheckList tests. The test sets stored in JSON files are partially hand-crafted and partially created using ChatGPT4.

## Pre-experimental Setup
- Ensure all dependencies in `requirements.txt` are installed.
- Download trained models from [Dropbox](https://www.dropbox.com/scl/fo/xv6pkmvqfs4eaptr0aw9i/h?rlkey=jk8ggqbkrngjclxduq2fod3dy&dl=0).
- Extract and place the models in the `models` directory.

## Files Overview
**Test Sets:**
- Seven key capabilities are assessed, with each represented in separate JSON files linked to specific SRL tests.

**BERT Input Files:**
- Test instances are pre-organized in the directories `input_M1`, `input_M2`, and `input_M3` for each model variant, formatted in CONLLU files ready for predictions.

**BERT Output Files:**
- Model predictions for each test type are stored in corresponding `output_M1`, `output_M2`, and `output_M3` directories.

## Scripts
**Preprocessing:**
- `preprocessing.py`: transforms the JSON test sets into CONLLU formatted files suitable for DistilBERT model predictions. This script ensures that the test instances are properly formatted and stored in separate files for each unique test type within their respective model input directories.

**Prediction:**
- `prediction.py`: executes all three fine-tuned DistilBERT models on their respective input CONLLU files. It writes the predictions in TSV format and organizes these files into the appropriate output directories for each model. This script ensures predictions are generated and stored systematically for further evaluation.

**Evaluation:**
- `evaluation.py`: designed to assess the performance of each model's predictions against the gold annotations. It uses three distinct evaluation functions to assess the models on Minimum Functionality Tests (MFT), Invariance Tests (INV), and Directionality Tests (DIR) separately. The evaluation output includes the failure rate and the sentence IDs of failed test instances for each file. The script automatically recognizes the test type contained within each file based on the file name (MFT/INV/DIR) and evaluates the test accordingly.







