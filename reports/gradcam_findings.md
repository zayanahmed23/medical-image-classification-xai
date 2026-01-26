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
 **Pattern:** In true positive cases, Grad-CAM predominantly highlights lung regions, with varying degrees of localization, and shows minimal reliance on image borders or non-anatomical artifacts.


### False Positives (NORMAL → PNEUMONIA)
- fp_artifact_border_01.png: Heatmap focuses on non-lung regions (borders/diaphragm), not on lung parenchyma.
- fp_lung_focus_02.png: Heatmap focuses on lung regions despite the absence of pneumonia, leading to an incorrect prediction.
 **Pattern:** False positive predictions arise from mixed explanation behaviors, including reliance on non-lung artifacts (e.g., borders or diaphragm) as well as lung-focused attention in the absence of true pathology.


### Random Samples
- random_typical_01.png: Heatmap shows broad lung-focused attention with minimal border influence.
- random_variant_02.png: Heatmap shows asymmetric lung-focused attention with moderate diffusion.
 **Pattern:** In randomly selected samples, Grad-CAM generally exhibits lung-centered attention with noticeable variability in spatial distribution, reflecting typical model behavior without strong dominance of non-anatomical artifacts.


## Hypotheses (why this happens)
- H1: Shortcut learning / artifacts (text markers, borders, contrast)
- H2: Dataset bias + imbalance pushes model toward pneumonia predictions
- H3: Grad-CAM can look plausible even when decision evidence is weak


## Next test (faithfulness)
Planned: Deletion test — mask top-k Grad-CAM region and measure confidence drop.
- Expected if faithful: masking important region reduces P(PNEUMONIA)
- Expected if unfaithful: confidence does not drop much


## Faithfulness Evaluation (Deletion Test)
### Results and Interpretation
- A deletion-based faithfulness test was conducted by removing the top 25% most salient pixels identified by Grad-CAM and observing the resulting change in model confidence.
- For true positive pneumonia predictions, confidence drops were minimal (approximately 1–2%), indicating that the highlighted regions were not strongly causal to the model’s decision. This suggests that the classifier may rely on distributed or non-salient features rather than the localized regions emphasized by Grad-CAM.
- In contrast, false positive cases exhibited substantial confidence drops (approximately 25–67%), demonstrating that Grad-CAM successfully identified spurious regions driving incorrect predictions. Removing these regions significantly weakened the model’s confidence, revealing artifact-driven decision behavior.
- Random samples showed inconsistent responses, including cases where confidence increased after deletion, further indicating instability and limited causal alignment of the explanations.
- Overall, these results suggest that while Grad-CAM can effectively expose misleading features in erroneous predictions, its faithfulness for correct predictions remains limited. This highlights the importance of complementing qualitative visualizations with quantitative faithfulness evaluations in medical imaging applications.


## Limitations and Future Work
This study focused on evaluating Grad-CAM explanations using a deletion-based faithfulness test at a fixed deletion ratio. Several extensions could further strengthen the analysis.
First, evaluating faithfulness across multiple deletion levels (e.g., 50%) would help assess the stability of confidence degradation trends and provide a more granular view of explanation robustness.
Second, incorporating a random-mask deletion baseline would enable stronger causal comparison by distinguishing meaningful explanation-driven effects from random information removal.
Third, complementary faithfulness metrics such as insertion tests could be applied to validate whether progressively adding salient regions increases model confidence in a consistent manner.
Finally, comparing Grad-CAM with alternative explainability methods could help determine whether the observed limitations are method-specific or indicative of broader explainability challenges in medical imaging models.
