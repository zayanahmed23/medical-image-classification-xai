# Explainable AI for Pneumonia Detection in Chest X-rays

## Overview
This project investigates the faithfulness of Grad-CAM explanations for a deep learning–based pneumonia classifier trained on chest X-ray images. Beyond standard performance evaluation, the focus is on understanding **when and why Grad-CAM explanations are reliable or misleading**, particularly in the presence of false positives.

The study combines qualitative visualization with a quantitative deletion-based faithfulness test to assess whether highlighted regions are causally important to the model’s predictions.


## Problem Statement
Deep learning models achieve high accuracy in medical imaging tasks, but their clinical adoption depends on trustworthy explanations. Grad-CAM is widely used for visual interpretability, yet its **faithfulness** whether highlighted regions truly influence model decisions—remains unclear.


This project addresses the question:
> *Do Grad-CAM explanations meaningfully reflect the decision-making process of a pneumonia classifier?*


## Dataset
- **Chest X-ray Images (Pneumonia)** – Kaggle  
- Binary classification: **NORMAL vs PNEUMONIA**
- Evaluation performed on the **test split**
The dataset follows the standard train/val/test split and is organized
in ImageFolder-compatible format.
Due to data size and licensing, images are not included in this repository.


## Model
- Architecture: **ResNet-18 (pretrained)**
- Fine-tuned for binary classification
- Best checkpoint saved based on validation performance

**Test performance:**
- Accuracy: **0.83**
- AUROC: **0.95**
- Dominant error mode: **NORMAL → PNEUMONIA false positives**


## Explainability Method
- **Grad-CAM** applied to the final convolutional block (`layer4[-1]`)
- Visual analysis conducted on:
  - True Positives
  - False Positives
  - Randomly selected samples

Qualitative patterns were extracted to identify common explanation behaviors and potential failure modes.

📄 Detailed qualitative analysis:  
➡️ `reports/gradcam_findings.md`


## Faithfulness Evaluation (Deletion Test)
To evaluate explanation faithfulness, a **deletion-based test** was conducted:

- The top **25% most salient pixels** identified by Grad-CAM were masked.
- Model confidence before and after deletion was compared.
- A larger confidence drop indicates higher faithfulness.


### Key Findings
- **True Positives:** Minimal confidence drop (~1–2%), suggesting weak causal alignment.
- **False Positives:** Large confidence drops (25–67%), indicating reliance on spurious or artifact-driven regions.
- **Random Samples:** Highly unstable behavior, including confidence increases after deletion.

These results demonstrate that **Grad-CAM is more effective at exposing erroneous decision mechanisms than explaining correct predictions**.


## Key Research Insights
- Grad-CAM explanations can appear visually plausible even when they are not causally meaningful.
- False positive predictions are often driven by identifiable spurious features.
- Quantitative faithfulness tests are essential to complement qualitative visualizations in medical XAI.


## Project Structure
    src/
        train.py
        evaluate.py
        explainability.py
        faithfulness_deletion.py
        utils.py

    results/
        checkpoints/
        metrics/
        visualizations/

    reports/
        gradcam_findings.md


## Future Work
- Evaluate deletion faithfulness at multiple masking levels
- Introduce random-mask baselines for causal comparison
- Compare Grad-CAM with alternative explainability methods


## Research Contributions
- Implemented a pneumonia classifier using transfer learning
- Performed structured qualitative analysis of Grad-CAM explanations
- Designed and executed a deletion-based faithfulness test
- Identified limitations of Grad-CAM in correct medical predictions


## Motivation
This project was conducted to gain hands-on research experience in medical imaging and explainable AI, with an emphasis on critical evaluation of interpretability methods rather than surface-level visualization.


## Author
Zayan Ahmed 
Background: Data Science / Machine Learning  
Research Interests: Explainable AI, Model Robustness, Machine Learning, Artificial Intelligence
