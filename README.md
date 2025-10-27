# Deep Learning and Application (EEE598)

### Arizona State University — Fall 2024  
**Instructor:** Dr. Suren Jayasuriya  
**Student:** Chih-Hao (Andy) Tsai

---

## Overview
This repository presents my coursework, experiments, and final project completed in **EEE598: Deep Learning and Application** at Arizona State University.  
All deep learning experiments were conducted using **ASU’s SOL high-performance computing (HPC) cluster**, leveraging **NVIDIA A100 GPUs** for large-scale model training and experimentation.  

Each assignment demonstrates key concepts in **machine learning and computer vision**, including **supervised learning, backpropagation, CNNs, transformers, and generative AI** — culminating in my final research project: *Handwriting Company Logos Recognition*.



---

## Table of Contents
- [Assignment 1: Introduction to Deep Learning](#assignment-1-introduction-to-deep-learning)
- [Assignment 2: Perceptrons, Backpropagation, and KAN](#assignment-2-perceptrons-backpropagation-and-kan)
- [Assignment 3: Vision-Based Deep Learning Models](#assignment-3-vision-based-deep-learning-models)
- [Assignment 4: Sequence Models, Transformers, and Generative AI](#assignment-4-sequence-models-transformers-and-generative-ai)
- [Final Project: Handwriting Company Logos Recognition](#final-project-handwriting-company-logos-recognition)
- [Poster Presentation](#poster-presentation)

---

## Assignment 1: Introduction to Deep Learning
**Key topics:** Dataset collection, model optimization, environment setup

- Collected and preprocessed a **custom image dataset (cats, dogs, bears)** with >200 images.
- Trained an image classifier using **Teachable Machine**, tuning parameters (epochs, batch size, learning rate).
- Configured a **Conda + PyTorch environment** and verified GPU allocation on **NVIDIA A100 (SOL)**.
- Implemented tensor operations, gradient computation, and network parameter analysis.

📘 *Skills:* Tensor basics · Autograd · GPU utilization · HPC (ASU SOL)
📂 *Report:* [`Assignment_1.pdf`](./HW/HW1/Assignment_1.pdf)

---

## Assignment 2: Perceptrons, Backpropagation, and KAN
**Key topics:** Neural network fundamentals, backpropagation, model comparison

- Implemented the **Perceptron algorithm** from scratch (with modified bias update) and visualized hyperplanes.
- Extended the dataset to demonstrate **non-linear inseparability**.
- Derived and coded the **backpropagation algorithm** manually using matrix calculus.
- Compared **MLP vs. Kolmogorov–Arnold Networks (KAN)** — analyzing training accuracy, loss, and runtime.

📘 *Skills:* Linear classifiers · Gradient derivation · MLP vs. KAN architecture  
📂 *Report:* [`Assignment_2.pdf`](./HW/HW2/Assignment_2.pdf)

---

## Assignment 3: Vision-Based Deep Learning Models
**Key topics:** CNN architectures · feature extraction · transfer learning

- Explored **VGG-16** intermediate layers for feature extraction and **perceptual loss** analysis.  
- Summarized and trained **EfficientNetV2-S** on the *Oxford Flowers-102* dataset, evaluating model accuracy and convergence behavior.  
- Implemented **multi-GPU training** on the **ASU SOL HPC cluster (NVIDIA A100)** to accelerate training and benchmarking.  
- Designed a **custom ResNet-36** architecture with an additional convolutional layer and a novel activation function to enhance ImageNet classification performance.  
- Applied **Grad-CAM** visualization and developed a **custom image-augmentation method** (randomized noise masking) to analyze robustness.

📘 *Skills:* CNNs · Feature visualization · Grad-CAM · Multi-GPU training · Custom architectures · HPC (ASU SOL)
📂 *Report:* [`Assignment_3.pdf`](./HW/HW3/Assignment_3.pdf)

---

## Assignment 4: Sequence Models, Transformers, and Generative AI
**Key topics:** GRUs, SIREN networks, Vision Transformers, GANs

- **Music Generation:** Trained a 2-layer GRU model using the MAESTRO dataset to generate piano sequences.  
- **Positional Encoding:** Reimplemented **SIREN** architecture and proposed a new **cosine-composite activation function** achieving faster convergence.  
- **Vision Transformer (Segmentation):** Applied face-parsing and depth-estimation models from Hugging Face to segment and stylize images.  
- **DCGAN:** Built and trained a **DCGAN** on both synthetic and real datasets (colored squares, cats faces), iteratively tuning parameters for realistic outputs.

📘 *Skills:* RNNs · Transformer-based segmentation · GAN training  
📂 *Report:* [`Assignment_4.pdf`](./HW/HW4/Assignment_4.pdf)

---

## Final Project: Handwriting Company Logos Recognition
**Key topics:** CNNs · Similarity search · GUI development

This capstone project develops an **image recognition system** that identifies **hand-drawn company logos** using multiple deep learning models.

### Highlights
- Built a **Tkinter GUI** allowing users to draw logos and perform real-time similarity search.
- Implemented and compared **AlexNet, ResNet-50, EfficientNet-B0, Swin Transformer,** and a **custom CNN Autoencoder**.
- Designed a **feature extraction + cosine similarity pipeline** for logo retrieval.
- Achieved up to **60% test accuracy** using EfficientNet-B0 with data augmentation.

📘 *Skills:* Autoencoders · Vision Transformers · GUI programming · Cosine similarity search  
📂 *Project Paper:* [`Final_Paper.pdf`](./FinalProject/Final_Paper.pdf)

---

## Poster Presentation
Summarized the final project workflow, comparative results, and future work (Vision Transformer extension).  
📂 *Poster:* [`FinalPoster.pdf`](./FinalProject/FinalPoster.pdf)

---

## Tools & Libraries
- **Frameworks:** PyTorch, TensorFlow, Hugging Face, Matplotlib  
- **Hardware / HPC:** NVIDIA A100 GPUs on ASU SOL Cluster 
- **Languages:** Python 3.10  
- **Others:** Tkinter, NumPy, Pandas, Google Colab


