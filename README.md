# Generative Adversarial Network (GAN) for Synthetic Image Generation

A deep learning project implementing a **Generative Adversarial Network (GAN)** from first principles to synthesize novel data, focusing on unsupervised learning of the underlying distribution of the image dataset.

---

## Project Title & Short Description

**Title:** Generative Adversarial Network (GAN) for Synthetic Image Generation (Keras Implementation)

**Description:** This project constructs a two-part adversarial network—a Generator and a Discriminator—using the **Keras/TensorFlow** framework. The goal is to train the Generator to create high-fidelity, never-before-seen synthetic images (using the **MNIST** dataset as a domain example).

---

## Problem Statement / Goal

The primary goal is to develop and stabilize a **Generative Adversarial Network (GAN) architecture** capable of unsupervised learning and data synthesis. This project aims to demonstrate the core principles of adversarial training where:
1.  The **Discriminator** learns to distinguish between real and synthetic data.
2.  The **Generator** learns to produce synthetic data convincing enough to fool the Discriminator.
This results in a model that can autonomously generate new, unique images that share the characteristics of the training set.

---

## Tech Stack / Tools Used

The solution is a deep learning project built using the Keras API, a high-level framework for TensorFlow:

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | Keras / TensorFlow | Core framework for defining, compiling, and training the GAN models |
| **Data Handling** | NumPy | Numerical operations and handling high-dimensional tensors (noise vectors) |
| **Data** | Keras Datasets (MNIST) | Source dataset for training the generative model |
| **Visualization**| Matplotlib | Visualizing the synthetic images generated during the training process |

---

## Approach / Methodology

1.  **Data Preprocessing**: Loaded the **MNIST handwritten digit dataset** and normalized the image pixels to the range **[-1, 1]** to match the output activation (e.g., tanh) of the Generator's final layer.
2.  **Architecture Definition**:
    * **Generator**: Defined a sequential network that accepts a random noise vector (latent space) and upsamples it through `Dense` and `Reshape` layers to output a 28x28x1 image.
    * **Discriminator**: Defined a sequential network that accepts a 28x28x1 image, flattens it, and outputs a single probability (real or fake).
3.  **Adversarial Model Assembly (GAN)**: The Generator and Discriminator were chained into a single adversarial model (`gan`) for the Generator's training phase.
4.  **Training Loop**: Implemented a custom training function (`train_gan`) that iteratively:
    * Feeds real images and generated fake images to the Discriminator.
    * Feeds random noise (with a target label of "real") to the GAN model, updating only the Generator's weights.
5.  **Visualization**: A function (`save_images`) was implemented to periodically sample the Generator and save 5x5 grids of synthetic images to track the evolution of the image quality.

---

## Results / Key Findings

* The training loop successfully demonstrated the adversarial process, with the Discriminator's accuracy oscillating while the Generator's loss improved over thousands of epochs.
* The final trained Generator is capable of synthesizing novel, unique handwritten digit images that were not present in the original dataset.
* The visualization step confirms the model's ability to learn and reproduce the complex spatial features of the MNIST digits.

---

## Topic Tags

GenerativeAI GAN DeepLearning Keras TensorFlow MNIST ImageGeneration SyntheticData AdversarialNetworks Python

---

## How to Run the Project

### 1. Install Requirements

Install all necessary packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
