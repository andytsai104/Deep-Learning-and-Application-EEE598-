# Deep Learning and Application (EEE598)

### Arizona State University — Fall 2024  
**Instructor:** Dr. Suren Jayasuriya  
**Student:** Chih-Hao (Andy) Tsai

---

## 🧠 Overview
This repository showcases my coursework, experiments, and final project completed in **EEE598: Deep Learning and Application** at Arizona State University.  
Each assignment demonstrates key machine learning and computer vision concepts, covering **supervised learning, backpropagation, CNNs, transformers, and generative AI** — culminating in my final research project: *Handwriting Company Logos Recognition*.

---

## 📑 Table of Contents
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

📘 *Skills:* Tensor basics · Autograd · GPU utilization  
📂 *Report:* [`Assignment_1.pdf`](./Assignment_1.pdf)

---

## Assignment 2: Perceptrons, Backpropagation, and KAN
**Key topics:** Neural network fundamentals, backpropagation, model comparison

- Implemented the **Perceptron algorithm** from scratch (with modified bias update) and visualized hyperplanes.
- Extended the dataset to demonstrate **non-linear inseparability**.
- Derived and coded the **backpropagation algorithm** manually using matrix calculus.
- Compared **MLP vs. Kolmogorov–Arnold Networks (KAN)** — analyzing training accuracy, loss, and runtime.

📘 *Skills:* Linear classifiers · Gradient derivation · MLP vs. KAN architecture  
📂 *Report:* [`Assignment_2.pdf`](./Assignment_2.pdf)

---

## Assignment 3: Vision-Based Deep Learning Models
**Key topics:** CNN architectures, feature extraction, transfer learning

- Explored **VGG-16** layers for feature extraction and perceptual loss.
- Summarized and trained **EfficientNetV2-S** on *Oxford Flowers-102* dataset (tested on dual A100 GPUs).
- Designed a **ResNet-36** architecture with a custom activation function to improve ImageNet performance.
- Implemented **Grad-CAM** visualization and custom image augmentation (noise masking).

📘 *Skills:* CNNs · Feature visualization · Grad-CAM · Multi-GPU training  
📂 *Report:* [`Assignment_3.pdf`](./Assignment_3.pdf)

---

## Assignment 4: Sequence Models, Transformers, and Generative AI
**Key topics:** GRUs, SIREN networks, Vision Transformers, GANs

- **Music Generation:** Trained a 2-layer GRU model using the MAESTRO dataset to generate piano sequences.  
- **Positional Encoding:** Reimplemented **SIREN** architecture and proposed a new **cosine-composite activation function** achieving faster convergence.  
- **Vision Transformer (Segmentation):** Applied face-parsing and depth-estimation models from Hugging Face to segment and stylize images.  
- **DCGAN:** Built and trained a **DCGAN** on both synthetic and real datasets (colored squares, cats faces), iteratively tuning parameters for realistic outputs.

📘 *Skills:* RNNs · Transformer-based segmentation · GAN training  
📂 *Report:* [`Assignment_4.pdf`](./Assignment_4.pdf)

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
📂 *Project Paper:* [`Final_Paper.pdf`](./Final_Paper.pdf)

---

## Poster Presentation
Summarized the final project workflow, comparative results, and future work (Vision Transformer extension).  
📂 *Poster:* [`FinalPoster.pdf`](./FinalPoster.pdf)

---

## 🧩 Tools & Libraries
- **Frameworks:** PyTorch, TensorFlow, Hugging Face, Matplotlib  
- **Hardware:** NVIDIA A100 GPU (ASU SOL)  
- **Languages:** Python 3.10  
- **Others:** Tkinter, NumPy, Pandas, Google Colab

---

## 📚 Course Context
**EEE598: Deep Learning and Application**  
Part of the *M.S. Robotics and Autonomous Systems – Electrical Engineering concentration* at **Arizona State University**.  
This course provided hands-on experience in **neural architectures, model design, and GPU-accelerated training** across vision and generative domains.

---

## 🔗 Related Links
- [ASU Robotics & Autonomous Systems (RAS) Program](https://ras.engineering.asu.edu/)
- [Course Handbook (MS-RAS 2024-2025)](./RAS-Handbook-24-25.pdf)
- [Graduate Portfolio Guidelines (EE Concentration)](./Graduate-MS-RAS-EE-Portfolio-Instructions.pdf)

---

## 🧾 License
This repository and its contents are for academic purposes only.  
All project codes, models, and reports were developed by **Chih-Hao (Andy) Tsai** under the supervision of **Dr. Suren Jayasuriya**.

