# SVHN-V1-ResNet

SVHN-V1-ResNet is a 21M parameter ResNet model trained on the SVHN dataset. Empirically, this method achieves a train loss of 98.90% and a test loss of 97.81%. SVHN-V1-ResNet shows strong performance across multiple other metrics.

## Evaluation

**F1 Score (Macro):** 97.71%

**F1 Score (Micro):** 97.81%

**Test Precision (Macro):** 97.74%

**Test Recall (Macro):** 97.69%

**Test Precision (Micro):** 97.81%

**Test Recall (Micro):** 97.81%

**ECE:** 0.085

**MCE:** 0.24

**Brier Score:** 0.043

**Prediction Mean Entropy:** 0.55

**Mean Confidence:** 0.89

**AUROC (OVR):** 99.77%

**AUPRC (Macro):** 99.05%

*The provided image shows a comprehensive analysis of the model's performance during training.*

<img width="8221" height="6056" alt="results" src="https://github.com/user-attachments/assets/37f0a0cb-682a-439e-a558-7412472946a1" />

## Technical Specifics

**Epochs:** 15

**Parameters:** ~21M

**Architecture:** ResNet-34

**Optimizer:** AdamW

**Loss Function:** Cross Entropy

**Learning Rate:** 0.001

**Batch Size:** 128

**Mean Grad Norm (Final):** ~0.005

Overall, this experimental model demonstrates ResNet's strong performance in computer vision tasks, particularly when paired with ReLU.
