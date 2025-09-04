# CMAKD: Cross-Modality Anatomy Knowledge Distillation

> **Status:** This repository contains the official implementation of our paper *"Cross-Modality Anatomy Knowledge Distillation for Retinal OCT Layer Segmentation with Limited Labels"* (under review).

---

## 📖 Introduction
Automatic segmentation of retinal Optical Coherence Tomography (OCT) images is essential for the early diagnosis and treatment of ocular diseases.  
However, labeled OCT data are often scarce, which limits the performance of supervised learning methods.  

In this work, we introduce **CMAKD**, a novel cross-modality knowledge distillation framework that leverages **abundant annotated Color Fundus Photography (CFP) data** to guide the segmentation of **layered retinal structures in OCT images**.  
Our method includes:

- **Domain-invariant Contrastive Learning (DoCL):** Bridges the modality gap by aligning CFP and OCT feature distributions.  
- **Layer-aware Self-ensembling Mean-Teacher (LA-MT):** Exploits unlabeled OCT images and enhances boundary representation of retinal layers.

Extensive experiments on three public datasets (Duke DME, MS, and GOALS) demonstrate that CMAKD significantly improves segmentation performance while reducing the reliance on OCT annotations.

---

## 📊 Results (Summary)
| Dataset  | Setting          | Improvement |
|----------|------------------|-------------|
| Duke DME | Semi-supervised  | **+17.8%** |
| MS       | Semi-supervised  | **+8.2%**  |
| GOALS    | Semi-supervised  | Approaches fully-supervised |

---




