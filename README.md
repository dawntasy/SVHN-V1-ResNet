[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/dawntasy/SVHN-V1-ResNet)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model-20beff?logo=kaggle&logoColor=white)](https://www.kaggle.com/models/learnwaterflow/svhn-v1-resnet)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Dawntasy/SVHN-V1-ResNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Usage

This model is a custom ResNet-34 architecture trained to recognize digits ($0-9$) from the **SVHN (Street View House Numbers)** dataset. Because it uses a custom implementation, you must set `trust_remote_code=True` when loading it.

Here are the three primary ways to use this model.

---

### Prerequisites
Ensure you have the necessary libraries installed:
```bash
pip install transformers torch pillow
```

---

### Method 1
This is the recommended way if you just want to get predictions. The `pipeline` handles image resizing, normalization, and mapping the output numbers back to digit labels (e.g., "3") automatically.

```python
from transformers import pipeline

# Load the classification pipeline
pipe = pipeline(
    "image-classification", 
    model="Dawntasy/SVHN-V1-ResNet", 
    trust_remote_code=True
)

# Predict using a URL or a local path
results = pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")

# Print the top prediction
print(f"Predicted Digit: {results[0]['label']} (Confidence: {results[0]['score']:.4f})")
```

---

### Method 2
Use this method if you need more control, such as running the model on a specific device (GPU/CPU) or processing batches of images. This separates the **preprocessing** from the **inference**.

```python
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

# 1. Load the processor and model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("Dawntasy/SVHN-V1-ResNet")
model = AutoModelForImageClassification.from_pretrained(
    "Dawntasy/SVHN-V1-ResNet", 
    trust_remote_code=True
).to(device)

# 2. Prepare the image
image = Image.open("your_digit_image.png").convert("RGB")
inputs = processor(image, return_tensors="pt").to(device)

# 3. Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# 4. Map index to label
print(f"Predicted Digit: {model.config.id2label[predicted_class_idx]}")
```

---

### Method 3
Use this method if you don't care about the final classification but want to use the model as a **feature extractor** (e.g., for image similarity, clustering, or as input to another model). This returns the 512-dimensional vector from the final global average pooling layer.

```python
from transformers import AutoModel, AutoImageProcessor
import torch

# Load the base "backbone" without the classification head
processor = AutoImageProcessor.from_pretrained("Dawntasy/SVHN-V1-ResNet")
base_model = AutoModel.from_pretrained(
    "Dawntasy/SVHN-V1-ResNet", 
    trust_remote_code=True
)

image = Image.open("your_digit_image.png").convert("RGB")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    # The output will be the flattened features before the final FC layer
    features = base_model(**inputs)
    
print(f"Feature vector shape: {features.logits.shape}") # torch.Size([1, 512])
```

---

### Input Requirements & Tips
*   **Image Size:** The model was trained on $64 \times 64$ pixel images. The `AutoImageProcessor` handles this resizing for you automatically.
*   **Normalization:** The model uses SVHN-specific mean and standard deviation: 
    *   Mean: `[0.4377, 0.4438, 0.4728]`
    *   Std: `[0.198, 0.201, 0.197]`
*   **Model Card Note:** This model is designed for digit recognition. When shown non-digit images (like animals or landscapes), it will still output a digit label based on the visual patterns that most resemble numbers.
