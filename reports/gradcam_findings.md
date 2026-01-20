# Grad-CAM Findings (Chest X-ray Pneumonia Classifier)

## Setup
- Dataset: Chest X-ray Images (Pneumonia) (Kaggle)
- Task: Binary classification (NORMAL vs PNEUMONIA)
- Model: ResNet-18 (pretrained), fine-tuned
- Checkpoint: results/checkpoints/model_best.pt
- Explainability: Grad-CAM on last conv block (layer4[-1])
- Splits analyzed: test
- Buckets: true positives, false positives, random (12 samples each)

## Quick performance context (from test_metrics.json)
- Accuracy: 0.8333333333333334  
- AUROC: 0.9524545255314486  
- Key error pattern: NORMAL misclassified as PNEUMONIA (false positives)

## Observations

### True Positives (PNEUMONIA → PNEUMONIA)
- tp_lung_localized_01.png: Heatmap focuses on localized lung regions, not on image borders or text artifacts.
- tp_lung_diffuse_02.png: Heatmap shows diffuse attention across lung fields, not on non-anatomical regions.
**Pattern:** _______________________________

### False Positives (NORMAL → PNEUMONIA)
- fp_artifact_border_01.png: Heatmap focuses on non-lung regions (borders/diaphragm), not on lung parenchyma.
- fp_lung_focus_02.png: Heatmap focuses on lung regions despite the absence of pneumonia, leading to an incorrect prediction.
**Pattern:** _______________________________

### Random Samples
- random_typical_01.png: Heatmap shows broad lung-focused attention with minimal border influence.
- random_variant_02.png: Heatmap shows asymmetric lung-focused attention with moderate diffusion.


## Hypotheses (why this happens)
- H1: Shortcut learning / artifacts (text markers, borders, contrast)
- H2: Dataset bias + imbalance pushes model toward pneumonia predictions
- H3: Grad-CAM can look plausible even when decision evidence is weak

## Next test (faithfulness)
Planned: Deletion test — mask top-k Grad-CAM region and measure confidence drop.
- Expected if faithful: masking important region reduces P(PNEUMONIA)
- Expected if unfaithful: confidence does not drop much
