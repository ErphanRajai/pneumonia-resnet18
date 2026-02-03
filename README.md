# The Lung Vision Project: Transparent AI for Pneumonia Detection

### The Origin Story

This project began as a personal quest to understand how deep learning can assist in critical medical decision-making. As someone fascinated by the intersection of healthcare and technology, I wanted to move beyond simple "black-box" classifiers.

I set out to build a tool for my portfolio that doesn't just provide a diagnosis, but acts as a **visual partner** for healthcare providers—highlighting exactly what the model sees when it flags a potential case of pneumonia.

### The Architecture: ResNet-18

To power this vision, I chose **ResNet-18**. In a medical environment, speed and reliability are paramount. ResNet-18’s residual skip-connections allowed for deep feature extraction without the risk of vanishing gradients, making it a "lean and mean" engine capable of running efficiently on standard hardware while maintaining high-tier accuracy.

### Proving the Concept: The Scores

When lives are on the line, **Recall** is the most important metric. We need to ensure that almost no sick patient goes undetected.

| Metric | Score | Impact |
| --- | --- | --- |
| **Recall (Pneumonia)** | **97.9%** | **Crucial:** We minimize the risk of missing a diagnosis. |
| **Accuracy** | **94.6%** | High overall reliability for clinical screening. |
| **Precision** | **92.3%** | Efficiently reduces the workload by minimizing false alarms. |

### Seeing Through the AI: Grad-CAM

The heart of this project is **explainability**. Using **Grad-CAM** (Gradient-weighted Class Activation Mapping), the app generates heatmaps over the chest X-rays.

* **The Point:** It’s not enough for the AI to say "Pneumonia." Grad-CAM allows the user to see *where* the model found the opacity or inflammation. This bridge of transparency is what builds trust between human expertise and machine intelligence.

### The Challenge: Balancing the Scales

One major hurdle was the **class imbalance**—a common issue in medical datasets where healthy images often outweigh sick ones (or vice versa). To solve this, I implemented custom data augmentation and weighted loss functions to ensure the model didn't become "lazy" and just guess the majority class.

---

### Launch the Project

You can explore the live model right now on **Hugging Face Spaces**:
👉 **[Live Demo Here](https://huggingface.co/spaces/itserphan/pneumonia-detection-resnet18)**

To run this locally on any device:

```bash
git clone https://github.com/ErphanRajai/pneumonia-resnet18.git
cd pneumonia-resnet18

pip install -r requirements.txt

# 3. Fire up the Streamlit interface
# This will open the app in your default web browser
streamlit run app.py

```