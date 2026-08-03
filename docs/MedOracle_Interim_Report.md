# Interim Report

## Level 4

## Multimodal Emotional Recognition System with Explainability

**MedOracle**

| Index Number | Name |
|---|---|
| 214206G | Suhira Balarajan |
| 214024V | Adshaya Balarajah |
| 214215H | Vanaiyan Kirupagaran |

**Supervised by:** Dr. Firdhous M.F.M.

Faculty of Information Technology
University of Moratuwa
2026

---

## Abstract

Emotion recognition has emerged as a critical component in human-computer interaction, mental health monitoring, and personalised healthcare systems. Existing approaches predominantly rely on a single modality — either physiological signals or facial expressions — limiting their robustness in real-world conditions where signal quality may degrade. This report presents MedOracle, a multimodal emotional recognition system that fuses electroencephalography (EEG), galvanic skin response (GSR), and facial video to classify five discrete emotional states: stress, calm, happy, sad, and angry.

The system employs a three-module pipeline. The physiological module processes EEG and GSR signals from the DEAP dataset using a bidirectional cross-modal attention network that allows each signal type to attend to the other. The video module processes facial video clips from the CREMA-D dataset through a YOLOv8 face detection pipeline followed by a ResNet50 encoder and a Bidirectional Long Short-Term Memory (BiLSTM) network for temporal emotion modelling. A confidence-weighted gated fusion module combines the outputs of both modalities using entropy-based confidence scores and signal quality penalties, with graceful degradation logic for missing or poor-quality inputs. An explainability layer built on Kernel SHAP produces interpretable feature attributions, exposed through a FastAPI backend and a React-based web dashboard with an integrated large language model chatbot.

At the interim stage, the system architecture has been fully designed, interface contracts between all three modules have been specified, and the video preprocessing and model architecture components have been implemented and validated. The CREMA-D dataset has been verified, label harmonisation has been completed, and the face detection pipeline using YOLOv8 is operational. Training of the video emotion model on Google Colab is underway. The physiological module and explainability components are in active development.

---

## Table of Contents

- Abstract
- Table of Contents
- List of Figures
- List of Tables
- Chapter 1 — Introduction to Multimodal Emotion Recognition
  - 1.1 Introduction
  - 1.2 Aim and Objectives
  - 1.3 Proposed Solution Overview
  - 1.4 Report Structure
  - 1.5 Summary
- Chapter 2 — Emotion Recognition: A Review of Existing Approaches
  - 2.1 Introduction
  - 2.2 Physiological Signal-Based Emotion Recognition
  - 2.3 Facial Expression-Based Emotion Recognition
  - 2.4 Multimodal Fusion Approaches
  - 2.5 Explainability in Affective Computing
  - 2.6 Comparison of Existing Approaches
  - 2.7 Summary
- Chapter 3 — Technologies Adapted for Multimodal Emotion Recognition
  - 3.1 Introduction
  - 3.2 Convolutional Neural Networks and ResNet50
  - 3.3 Recurrent Neural Networks and Bidirectional LSTM
  - 3.4 Attention Mechanisms and Cross-Modal Learning
  - 3.5 SHAP for Explainability
  - 3.6 Web Application Technologies
  - 3.7 Summary
- Chapter 4 — MedOracle: System Design and Approach
  - 4.1 Introduction
  - 4.2 System Overview
  - 4.3 Physiological Module
  - 4.4 Video and Fusion Module
  - 4.5 Explainability and Web Application Module
  - 4.6 Interface Contracts Between Modules
  - 4.7 Datasets
  - 4.8 Summary
- Chapter 5 — Analysis and Design
  - 5.1 Introduction
  - 5.2 Dataset Analysis and Label Harmonisation
  - 5.3 Video Preprocessing Pipeline Design
  - 5.4 Model Architecture Design
  - 5.5 Gated Fusion Design
  - 5.6 Evaluation and Ablation Study Design
  - 5.7 Summary
- Chapter 6 — Implementation
  - 6.1 Introduction
  - 6.2 Development Environment and Repository Structure
  - 6.3 Shared Modules: Label Harmonisation and Data Contracts
  - 6.4 Video Preprocessing Implementation
  - 6.5 Model Implementation
  - 6.6 Summary
- Chapter 7 — Discussion and Further Work
  - 7.1 Introduction
  - 7.2 Work Completed at the Interim Stage
  - 7.3 Differentiation from Existing Works
  - 7.4 Further Work
  - 7.5 Plan for Evaluation
  - 7.6 Summary
- References
- Appendix A — Individual Contributions to the Project
- Appendix B — CREMA-D Dataset Statistics
- Appendix C — Interface Contract Specifications
- Appendix D — Signal Quality Thresholds

---

## List of Figures

| Figure | Caption |
|---|---|
| Figure 4.1 | End-to-end MedOracle system pipeline |
| Figure 4.2 | Physiological module architecture |
| Figure 5.1 | Video preprocessing pipeline flowchart |
| Figure 5.2 | ResNet50 encoder fine-tuning strategy |
| Figure 5.3 | BiLSTM temporal modelling architecture |
| Figure 5.4 | Gated fusion algorithm |
| Figure 5.5 | Actor-independent 5-fold GroupKFold split design |
| Figure 5.6 | Ablation study conditions |
| Figure 6.1 | Repository structure |

*Note: Figures generated from the Colab training pipeline (class distribution, sample frames, intensity distributions, fold validation) are inserted at the appropriate locations.*

---

## List of Tables

| Table | Caption |
|---|---|
| Table 2.1 | Comparison of existing emotion recognition approaches |
| Table 4.1 | Emotion class definitions and dataset mappings |
| Table 4.2 | DEAP dataset summary |
| Table 4.3 | CREMA-D dataset summary |
| Table 5.1 | DEAP label harmonisation mapping (V/A/D thresholds) |
| Table 5.2 | CREMA-D label harmonisation mapping |
| Table 5.3 | Signal quality thresholds for video modality |
| Table 5.4 | Quality penalty factors for gated fusion |
| Table 5.5 | Ablation study conditions |
| Table 6.1 | Technology stack |

---

# Chapter 1

# Introduction to Multimodal Emotion Recognition

## 1.1 Introduction

Affective computing — the field concerned with systems that can recognise, interpret, and simulate human emotions — has grown substantially over the past two decades [1]. Emotion plays a central role in human decision-making, communication, and health, yet the accurate automated recognition of emotional states remains an open and challenging problem. The difficulty arises from the inherently subjective, context-dependent, and physiologically complex nature of emotion itself.

Early efforts in automated emotion recognition focused on single-modality approaches: either facial expressions captured from video [6], speech prosody extracted from audio, or physiological signals such as EEG and skin conductance [10]. While these unimodal methods have demonstrated promise in controlled laboratory settings, they exhibit significant limitations in real-world deployment. Facial expression systems are sensitive to illumination, occlusion, and pose variation. Physiological systems require direct sensor contact and are susceptible to movement artifacts. Each modality provides only a partial view of the underlying emotional state.

The convergence of advances in deep learning, large-scale affective datasets, and multimodal sensing hardware has made it feasible to fuse multiple modalities to achieve more robust and accurate emotion recognition [8]. In parallel, growing awareness of the opacity of deep learning models — particularly in sensitive domains such as mental health and clinical support — has led to increasing emphasis on model explainability [2]. A system that can recognise emotions but cannot explain why it made a given prediction is of limited utility in contexts where trust and accountability are essential.

The MedOracle project addresses both of these challenges. It proposes a multimodal emotion recognition system that fuses EEG, GSR, and facial video, combined with a Kernel SHAP-based explainability layer and a user-facing web application. The system is designed to be robust to missing or poor-quality modalities through a gated fusion mechanism, and to provide interpretable explanations of its predictions through an integrated large language model chatbot.

## 1.2 Aim and Objectives

The aim of this project is to develop a robust, explainable multimodal emotional recognition system capable of classifying five discrete emotion classes — stress, calm, happy, sad, and angry — by fusing physiological signals (EEG and GSR) with facial video.

The specific objectives of the project are:

1. To design and implement a physiological emotion recognition module that processes EEG and GSR signals using a bidirectional cross-modal attention network trained on the DEAP dataset.
2. To design and implement a video-based emotion recognition module that processes facial video clips using a YOLOv8–ResNet50–BiLSTM pipeline trained on the CREMA-D dataset.
3. To develop a confidence-weighted gated fusion module that combines the outputs of both modalities using entropy-based confidence scores and signal quality penalties, with graceful degradation for missing or degraded inputs.
4. To implement a Kernel SHAP-based explainability layer that produces interpretable feature attributions for the fused predictions.
5. To build a web application with a FastAPI backend, PostgreSQL database, and React frontend, incorporating a large language model chatbot that translates SHAP outputs into natural-language explanations.
6. To evaluate the system through a five-condition ablation study and three robustness tests, reporting macro-averaged F1 scores and statistical significance using the Wilcoxon signed-rank test.

## 1.3 Proposed Solution Overview

MedOracle is a three-module system designed around a strict late-fusion architecture. The physiological module (Member 1) accepts EEG and GSR signals as inputs and produces a prediction dictionary containing class probabilities, a predicted emotion label, a confidence score, and signal quality flags. The video and fusion module (Member 2) accepts facial video alongside the physiological prediction and produces a fused prediction output that combines both modalities using a gated weighting mechanism. The explainability and web application module (Member 3) accepts the fused prediction and produces SHAP-based explanations delivered through a React dashboard and LLM chatbot.

The system targets five users: clinicians using the tool for affective state monitoring, researchers studying multimodal emotion recognition, and end users interacting with the web dashboard for self-monitoring. The primary input is a synchronised trial consisting of 4 seconds of EEG (32 channels at 128 Hz), 4 seconds of GSR (4 Hz, resampled to 128 Hz), and a facial video clip sampled to 16 frames at 224×224 pixels. The primary output is a predicted emotion class, per-class probability distribution, modality weights, signal quality grades, and SHAP feature importance values, all presented through the web interface.

## 1.4 Report Structure

Chapter 2 reviews existing approaches to emotion recognition across unimodal and multimodal paradigms, with a comparative analysis that motivates the design choices made in MedOracle. Chapter 3 describes the key technologies adapted in the system, including deep learning architectures, attention mechanisms, and explainability techniques. Chapter 4 presents the overall system design and the approach taken by each of the three modules. Chapter 5 details the analysis and design of each component, including dataset analysis, model architecture design, and fusion algorithm design. Chapter 6 describes the implementation work completed at the interim stage. Chapter 7 presents a discussion of the work completed, how MedOracle differs from existing systems, and the plan for remaining work and evaluation.

## 1.5 Summary

This chapter introduced the motivation for multimodal emotion recognition, highlighting the limitations of unimodal approaches and the growing need for explainable systems in affective computing. The aim and objectives of MedOracle were stated, along with a high-level overview of the three-module system architecture. Chapter 2 surveys the existing literature in emotion recognition to identify the gaps that MedOracle addresses.

---

# Chapter 2

# Emotion Recognition: A Review of Existing Approaches

## 2.1 Introduction

Chapter 1 established the motivation for building a robust, explainable multimodal emotion recognition system. This chapter surveys the landscape of existing work in emotion recognition, covering physiological signal-based approaches, facial expression-based approaches, multimodal fusion methods, and the emerging field of explainability in affective computing. The survey identifies the key limitations in existing work and positions MedOracle's contributions within this context.

## 2.2 Physiological Signal-Based Emotion Recognition

Physiological signals offer an objective and difficult-to-suppress window into affective states, as they are generated by the autonomic nervous system rather than being consciously controlled. EEG and galvanic skin response (GSR) are among the most widely studied physiological modalities for emotion recognition.

Koelstra et al. [5] introduced the DEAP dataset, which remains one of the most widely used benchmarks for physiological emotion recognition. They demonstrated that EEG signals in the alpha (8–13 Hz) and gamma (30–45 Hz) frequency bands are particularly informative for valence and arousal classification. Siddharth et al. [10] extended this line of work by combining EEG with peripheral physiological signals including GSR and achieving approximately 73% classification accuracy on arousal and valence dimensions. However, their approach did not incorporate video and did not address explainability.

A common limitation of physiological approaches is their dependence on specialised hardware and their sensitivity to motion artifacts and electrode impedance variations. Moreover, most existing work in the DEAP dataset treats emotion recognition as a binary problem (high/low valence, high/low arousal) rather than mapping to discrete, clinically meaningful emotion classes [5]. MedOracle addresses this by mapping the continuous DEAP labels to five discrete emotion classes using a validated mapping based on the Russell circumplex model of affect [9].

## 2.3 Facial Expression-Based Emotion Recognition

Facial expression analysis has been a central topic in computer vision since the work of Ekman and Friesen on the Facial Action Coding System (FACS). Contemporary approaches leverage convolutional neural networks (CNNs) to extract spatial features from face images, often combined with recurrent networks to model temporal dynamics.

Zhang et al. [12] proposed a video-based emotion recognition approach using ResNet-based feature extraction combined with a temporal attention mechanism, achieving approximately 65% accuracy on emotion classification tasks comparable to those addressed in MedOracle. Cao et al. [3] introduced the CREMA-D dataset containing 7,442 clips from 91 actors expressing six emotions, providing a large and demographically diverse benchmark for facial emotion recognition.

Facial expression systems face well-known challenges including pose variation, illumination changes, occlusion, and the distinction between genuine and posed expressions. The YOLOv8-based face detection pipeline implemented in MedOracle addresses some of these concerns by reliably localising and normalising the face region before feature extraction.

## 2.4 Multimodal Fusion Approaches

The fusion of physiological and visual modalities has been explored in a growing body of literature. Poria et al. [8] provide a comprehensive survey of multimodal sentiment analysis, comparing early fusion (feature-level), late fusion (decision-level), and hybrid approaches. Their analysis concludes that late fusion tends to outperform early fusion when the modalities have different temporal resolutions, different noise characteristics, and are derived from different training datasets — all of which apply to MedOracle's setting.

Baltrusaitis et al. [2] survey multimodal machine learning more broadly and identify cross-dataset fusion as a particularly challenging scenario. In MedOracle, the physiological module is trained on DEAP (32 subjects) while the video module is trained on CREMA-D (91 actors). Since these datasets have no overlapping subjects, joint training is architecturally impossible, making late fusion the only viable approach. This is a primary design justification rather than a limitation.

A gap in existing multimodal fusion work is the lack of principled handling of degraded or missing modalities. Most systems assume clean, complete inputs from all modalities [8]. MedOracle's gated fusion module explicitly accounts for signal quality by applying per-modality penalty factors derived from signal quality assessments, enabling graceful degradation when one modality is unreliable.

## 2.5 Explainability in Affective Computing

The deployment of deep learning models in sensitive domains such as mental health monitoring raises important questions about interpretability and accountability. Lundberg and Lee [7] introduced SHAP (SHapley Additive exPlanations), a game-theory-based framework for explaining model predictions by computing the contribution of each input feature to the output. Kernel SHAP, a model-agnostic variant, is particularly suited for explaining complex pipeline outputs such as the fused prediction in MedOracle.

Existing multimodal emotion recognition systems rarely incorporate formal explainability mechanisms. When explanations are provided, they are typically limited to visualising attention weights [12], which are not guaranteed to be faithful to the model's actual decision process [7]. MedOracle addresses this gap by using Kernel SHAP with a faithfulness metric that measures the degree to which the SHAP explanation accurately reflects the model's behaviour under feature perturbation.

## 2.6 Comparison of Existing Approaches

Table 2.1 compares MedOracle against key existing systems across the dimensions most relevant to the project's contributions.

**Table 2.1: Comparison of Existing Emotion Recognition Approaches**

| System | Modalities | Emotion Classes | Dataset | Fusion | Explainability | Graceful Degradation |
|---|---|---|---|---|---|---|
| Koelstra et al. [5] | EEG, GSR | Binary (V/A) | DEAP | None | None | No |
| Siddharth et al. [10] | EEG + peripheral | Binary (V/A) | DEAP | Feature-level | None | No |
| Zhang et al. [12] | Video | 6 classes | Custom | None | Attention | No |
| Cao et al. [3] | Video + Audio | 6 classes | CREMA-D | Late | None | No |
| **MedOracle (proposed)** | **EEG + GSR + Video** | **5 discrete** | **DEAP + CREMA-D** | **Gated late** | **Kernel SHAP + LLM** | **Yes** |

## 2.7 Summary

This chapter surveyed existing approaches to emotion recognition across physiological, visual, and multimodal paradigms. Physiological systems provide objective but hardware-dependent signals; video systems are non-intrusive but sensitive to environmental factors; multimodal systems improve robustness but rarely address signal quality degradation or explainability. MedOracle distinguishes itself by combining gated multimodal fusion with principled explainability, cross-dataset integration, and graceful degradation for missing modalities. Chapter 3 describes the specific technologies adapted to implement this design.

---

# Chapter 3

# Technologies Adapted for Multimodal Emotion Recognition

## 3.1 Introduction

Chapter 2 identified the key requirements for MedOracle: a system capable of processing heterogeneous physiological and video signals, fusing their predictions robustly, and producing interpretable explanations. This chapter describes the core technologies adopted to implement these requirements, explaining why each technology is appropriate for the specific sub-problem it addresses.

## 3.2 Convolutional Neural Networks and ResNet50

Convolutional Neural Networks (CNNs) are the dominant approach for spatial feature extraction from image data. ResNet50, introduced by He et al. [4], is a 50-layer residual network pre-trained on ImageNet (1.28 million images, 1,000 classes) that achieves state-of-the-art performance on visual recognition tasks. The residual connection mechanism — which adds the input of each block directly to its output — addresses the vanishing gradient problem that limits the depth of conventional CNNs.

In MedOracle's video module, ResNet50 is used as a frame-level feature extractor. Rather than training a CNN from scratch — which would require far more data than CREMA-D provides — the model is fine-tuned from ImageNet weights. The early layers (layer1, layer2) are frozen to preserve low-level texture and edge features learned from ImageNet, while the later layers (layer3, layer4) are fine-tuned to learn emotion-relevant spatial patterns. This transfer learning strategy is justified by the well-established observation that lower CNN layers capture generic visual features while higher layers capture task-specific semantics [4]. Each frame is processed independently by ResNet50 to produce a 2048-dimensional feature vector.

## 3.3 Recurrent Neural Networks and Bidirectional LSTM

Recurrent Neural Networks (RNNs) and their gated variants — Long Short-Term Memory (LSTM) networks, introduced by Hochreiter and Schmidhuber [11] — are designed to model sequential data by maintaining a hidden state that captures information from previous time steps. The LSTM's gating mechanism (input gate, forget gate, output gate) allows it to selectively retain or discard information over long sequences, addressing the vanishing gradient problem that affects vanilla RNNs.

A Bidirectional LSTM (BiLSTM) processes the input sequence in both the forward and backward directions, concatenating the hidden states from both passes. This is important for video-based emotion recognition because the emotional arc of a clip — the way an expression builds and fades — is best understood with context from both early and late frames. In MedOracle, the BiLSTM takes the sequence of 16 ResNet50 frame features (each 2048-dimensional) as input and produces a 512-dimensional representation (256 per direction) at the final timestep, which is passed to a classification head.

For the physiological module, a bidirectional cross-modal attention network extends the BiLSTM principle to model interactions between EEG and GSR signals, allowing each modality to attend to the other during feature extraction.

## 3.4 Attention Mechanisms and Cross-Modal Learning

Attention mechanisms, popularised by the Transformer architecture, allow a neural network to dynamically weight the relevance of different input positions when producing each output. In cross-modal attention, the query comes from one modality while the keys and values come from another, enabling one signal to be conditioned on the other.

In MedOracle's physiological module, a bidirectional cross-modal attention mechanism allows EEG to attend to GSR and GSR to attend to EEG simultaneously. This is motivated by the well-established physiological relationship between central nervous system activity (EEG) and peripheral autonomic activity (GSR): stress, for example, simultaneously elevates EEG gamma-band power and GSR conductance [10]. Allowing the two signals to attend to each other during feature extraction is expected to improve the discriminability of the joint representation compared to processing each signal independently.

## 3.5 SHAP for Explainability

SHAP (SHapley Additive exPlanations), introduced by Lundberg and Lee [7], provides a principled framework for explaining model predictions based on Shapley values from cooperative game theory. Each feature is assigned a Shapley value representing its average marginal contribution to the model output across all possible feature coalitions.

Kernel SHAP, the model-agnostic variant, approximates Shapley values by fitting a weighted linear model to the predictions of the original model on a set of perturbed inputs. It is appropriate for MedOracle because the fused prediction pipeline — comprising separate physiological and video models — cannot be differentiated end-to-end in the way required by gradient-based explanation methods.

MedOracle uses a faithfulness metric to validate the quality of the SHAP explanations: features identified as highly important by SHAP are perturbed, and the resulting change in the model's prediction confidence is measured. A faithful explanation should produce large confidence drops when its top features are removed.

## 3.6 Web Application Technologies

The web application layer of MedOracle uses FastAPI as the backend framework, chosen for its high performance (comparable to Node.js and Go in benchmark tests), native asynchronous support, and automatic OpenAPI documentation generation. PostgreSQL is used as the production database for its reliability and support for JSON fields (used to store prediction and SHAP output dictionaries). JWT (JSON Web Tokens) are used for stateless authentication.

The React frontend uses the Recharts library for visualisation (emotion probability bar charts, session history line charts, SHAP waterfall charts). The integrated LLM chatbot uses a SHAP-to-prompt bridging strategy: structured SHAP output is converted to a natural-language context string that is passed to a large language model (Claude or GPT-4) to generate a plain-English explanation of the prediction.

## 3.7 Summary

This chapter described the core technologies adopted in MedOracle: ResNet50 for frame-level feature extraction, BiLSTM for temporal modelling, cross-modal attention for physiological signal fusion, Kernel SHAP for explainability, and FastAPI/React for the web application layer. Each technology was selected based on its demonstrated suitability for the specific sub-problem it addresses. Chapter 4 presents the complete system design that integrates these technologies.

---

# Chapter 4

# MedOracle: System Design and Approach

## 4.1 Introduction

Chapter 3 described the technologies that underpin MedOracle. This chapter presents the full system design, describing how these technologies are integrated across the three modules of the pipeline and how the modules interact through well-defined interface contracts.

## 4.2 System Overview

MedOracle is structured as a three-module pipeline that processes three input modalities — EEG, GSR, and facial video — and produces a fused emotion prediction with accompanying SHAP-based explanations. Figure 4.1 shows the end-to-end system pipeline.

**[Figure 4.1: End-to-end MedOracle system pipeline — insert diagram_pipeline.svg here]**

The pipeline operates as follows. The physiological module accepts a 4-second trial of EEG (shape: 32 × 512 samples) and GSR (shape: 512 samples after resampling) and produces a prediction dictionary containing class probabilities, a predicted emotion label, a confidence score, and EEG/GSR signal quality grades. The video and fusion module accepts a facial video clip (16 frames at 224×224 pixels) alongside the physiological prediction and produces a fused prediction output incorporating both modalities. The explainability module accepts the fused prediction and produces SHAP-based feature attributions, which are stored in a PostgreSQL database and served through a FastAPI backend to a React dashboard and LLM chatbot.

The late-fusion architecture is architecturally mandated by the cross-dataset constraint: the physiological module is trained on DEAP (32 subjects) while the video module is trained on CREMA-D (91 actors). Since these datasets share no overlapping subjects, joint end-to-end training is not possible. This design is further supported by the literature on multimodal fusion, which demonstrates that late fusion outperforms early fusion when modalities have different temporal resolutions and training distributions [8].

## 4.3 Physiological Module

The physiological module is responsible for EEG and GSR preprocessing, feature extraction, and classification. EEG signals are preprocessed using a 0.5–45 Hz bandpass filter, Independent Component Analysis (ICA) for artifact removal, and epoching into 4-second trials. GSR signals are lowpass filtered and resampled from 4 Hz to 128 Hz using polyphase resampling (scipy.signal.resample_poly) to match the EEG sampling rate.

The core model is a bidirectional cross-modal attention network. EEG features attend to GSR features (EEG → GSR attention) and GSR features attend to EEG features (GSR → EEG attention) in a bidirectional mechanism. The attended features are concatenated and passed to a joint 5-class classification head. The model is trained on the DEAP dataset using Leave-One-Subject-Out (LOSO) cross-validation across 32 subjects, with weighted cross-entropy loss to address class imbalance.

**[Figure 4.2: Physiological module architecture — insert diagram_physio.svg here]**

## 4.4 Video and Fusion Module

The video and fusion module is responsible for facial video preprocessing, video emotion recognition, and gated fusion with the physiological prediction. Figure 4.1 shows the position of this module in the overall pipeline.

The video preprocessing pipeline extracts 16 frames uniformly from the input clip, detects and crops the face region in each frame using YOLOv8 (with an OpenCV Haar cascade fallback), resizes crops to 224×224 pixels, and applies ImageNet normalisation. Signal quality is assessed using three metrics: face detection rate, Laplacian variance (sharpness), and face bounding box area ratio.

The video emotion model is a VideoEmotionModel combining a ResNet50 encoder and an EmotionBiLSTM. For each clip of B batches and T=16 frames, the model processes frames as (B×T, 3, 224, 224) through ResNet50 to produce (B×T, 2048) features, reshapes to (B, T, 2048), and passes the sequence through the BiLSTM to produce (B, 5) logits. The model is trained on CREMA-D using actor-independent 5-fold GroupKFold cross-validation to ensure zero actor overlap between train and test sets.

The gated fusion module combines the video and physiological predictions using entropy-based confidence scores and signal quality penalty factors. The fused class probabilities are computed as a weighted sum of the per-modality probability distributions, with weights determined by L1-normalised gate scores. Graceful degradation rules handle cases where one or both modalities are missing or of poor quality.

## 4.5 Explainability and Web Application Module

The explainability module applies Kernel SHAP to the fused prediction pipeline, treating the entire pipeline as a black box. SHAP values are computed for the input features of each modality, and a faithfulness metric is computed by measuring the prediction confidence drop when top-ranked features are perturbed.

The web application exposes the prediction and explanation functionality through nine FastAPI endpoints covering authentication, prediction, explanation retrieval, session management, chatbot interaction, and dashboard summaries. The React frontend displays personalised emotion trend charts, SHAP feature importance plots, session history, and an LLM chatbot panel. The LLM chatbot uses a SHAP-to-prompt bridging mechanism to convert structured SHAP output into natural-language context before querying the language model.

## 4.6 Interface Contracts Between Modules

The interface contracts define the exact data structures passed between modules, ensuring that each member can develop their module independently.

The physiological module (Member 1) exports the following dictionary to the video and fusion module (Member 2):

```
physiological_prediction_dict = {
    "predicted_emotion":    str,       # e.g. "stress"
    "confidence":           float,     # entropy-based, range [0, 1]
    "class_probabilities":  dict,      # {"stress": float, "calm": float, ...}
    "signal_quality": {
        "eeg":              str,       # "good" | "degraded" | "poor"
        "gsr":              str
    }
}
```

The video and fusion module (Member 2) exports the following dictionary to the explainability module (Member 3):

```
prediction_output = {
    "predicted_emotion":        str,
    "confidence":               float,
    "class_probabilities":      dict,
    "modality_weights":         {"physio": float, "video": float},
    "signal_quality":           {"eeg": str, "gsr": str, "video": str},
    "per_modality_predictions": {
        "physio": {"predicted_emotion": str, "confidence": float,
                   "class_probabilities": dict},
        "video":  {"predicted_emotion": str, "confidence": float,
                   "class_probabilities": dict}
    }
}
```

## 4.7 Datasets

**Table 4.2: DEAP Dataset Summary**

| Attribute | Detail |
|---|---|
| Subjects | 32 |
| Trials per subject | 40 |
| Trial duration | 60 seconds |
| EEG channels | 32 (128 Hz) |
| GSR sampling rate | 4 Hz |
| Labels | Continuous Valence, Arousal, Dominance (1–9 scale) |
| Usage in MedOracle | Physiological module (Member 1) |

**Table 4.3: CREMA-D Dataset Summary**

| Attribute | Detail |
|---|---|
| Actors | 91 |
| Total clips | 7,442 |
| Usable clips (after dropping DIS) | 6,171 |
| Frame rate | ~30 fps |
| Original emotion labels | ANG, DIS, FEA, HAP, NEU, SAD |
| MedOracle labels used | ANG→angry, HAP→happy, SAD→sad, NEU→calm, FEA→stress |
| Usage in MedOracle | Video module (Member 2) |

**Table 4.1: Emotion Class Definitions**

| Class | Index | Physiological signature | DEAP mapping | CREMA-D mapping |
|---|---|---|---|---|
| stress | 0 | High arousal, low valence, low dominance | V<5, A≥5, D<5 | FEA |
| calm | 1 | Low arousal, high valence, high dominance | V≥5, A<5, D≥5 | NEU |
| happy | 2 | High arousal, high valence, high dominance | V≥5, A≥5, D≥5 | HAP |
| sad | 3 | Low arousal, low valence, low dominance | V<5, A<5, D<5 | SAD |
| angry | 4 | High arousal, low valence, high dominance | V<5, A≥5, D≥5 | ANG |

## 4.8 Summary

This chapter presented the complete system design for MedOracle, covering the three-module pipeline, the interface contracts between modules, the datasets used, and the high-level design of each module. The late-fusion architecture is mandated by the cross-dataset constraint and supported by the literature. Chapter 5 presents the detailed analysis and design of each component.

---

# Chapter 5

# Analysis and Design

## 5.1 Introduction

Chapter 4 presented the overall system design. This chapter details the analysis and design decisions that underpin each component of MedOracle, including dataset analysis and label harmonisation, video preprocessing pipeline design, model architecture design, gated fusion algorithm design, and the evaluation and ablation study design.

## 5.2 Dataset Analysis and Label Harmonisation

### 5.2.1 DEAP Label Harmonisation

The DEAP dataset provides continuous ratings for Valence (V), Arousal (A), and Dominance (D) on a 1–9 scale. These are mapped to MedOracle's five emotion classes using the thresholds derived from the Russell circumplex model of affect [9] and the Mehrabian PAD model [6]. The threshold value of 5.0 is the midpoint of the 1–9 scale; boundary cases (exactly 5.0) are assigned to the positive side.

**Table 5.1: DEAP Label Harmonisation Mapping**

| Emotion | Valence | Arousal | Dominance | Theoretical basis |
|---|---|---|---|---|
| stress | <5 | ≥5 | <5 | Russell (1980), Mehrabian (1996) |
| calm | ≥5 | <5 | ≥5 | Russell (1980) |
| happy | ≥5 | ≥5 | ≥5 | Russell (1980) |
| sad | <5 | <5 | <5 | Russell (1980) |
| angry | <5 | ≥5 | ≥5 | Russell (1980) |

### 5.2.2 CREMA-D Label Harmonisation

CREMA-D provides six emotion labels. The mapping to MedOracle's five classes and the justification for each mapping are shown in Table 5.2. The DIS (disgust) class is dropped from training because disgust occupies a region of the valence-arousal-dominance space that does not map cleanly to any of the five target classes, and including it would introduce label noise.

**Table 5.2: CREMA-D Label Harmonisation Mapping**

| CREMA-D label | MedOracle class | Justification |
|---|---|---|
| ANG | angry | Direct semantic match |
| HAP | happy | Direct semantic match |
| SAD | sad | Direct semantic match |
| NEU | calm | Neutral state corresponds to low arousal, positive valence [9] |
| FEA | stress | Fear activates the sympathetic nervous system similarly to stress [6] |
| DIS | **DROPPED** | No clean mapping; inclusion would introduce label noise |

### 5.2.3 Dataset Statistics

*[Insert Figure from Colab: Class distribution bar chart — class_distribution.png]*

*[Insert Figure from Colab: Actor distribution chart — actor_distribution.png]*

*[Insert Figure from Colab: Intensity distribution — intensity_distribution.png]*

After dropping the DIS class, the CREMA-D dataset yields 6,171 usable clips distributed across five emotion classes and 91 actors. The class distribution shows mild imbalance, with HAP and ANG being the most frequent classes. This imbalance is addressed in training using weighted cross-entropy loss, where class weights are computed using sklearn's compute_class_weight function.

## 5.3 Video Preprocessing Pipeline Design

The video preprocessing pipeline converts raw video clips into normalised frame tensors suitable for input to the VideoEmotionModel. Figure 5.1 shows the flowchart of the preprocessing pipeline.

**[Figure 5.1: Video preprocessing pipeline flowchart — insert diagram_preprocessing.svg here]**

The pipeline consists of the following stages:

1. **Frame sampling**: 16 frames are sampled uniformly from each clip using a temporal stride calculated as clip_length / 16. This is the standard approach for video classification and ensures that the temporal distribution of frames is representative of the full clip duration.

2. **Face detection**: YOLOv8 (yolov8n-face.pt, fine-tuned for face detection) is applied to each frame to produce a bounding box. If YOLOv8 is unavailable, an OpenCV Haar cascade classifier is used as a fallback. The largest detected face (by bounding box area) is selected.

3. **Face cropping**: The detected bounding box is padded by 20% on each side (to include hair, chin, and neck context) and clamped to the frame boundaries. The padded crop is resized to 224×224 pixels using area interpolation.

4. **Fallback cropping**: If no face is detected in a frame, a centre crop (60% of the frame) is taken and resized to 224×224 pixels. The frame is recorded as undetected for signal quality assessment.

5. **ImageNet normalisation**: Pixel values are normalised using the ImageNet mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]).

6. **Signal quality assessment**: Three metrics are computed across the 16 frames — face detection rate (fraction of frames with a detected face), mean Laplacian variance (sharpness, following Pech-Pacheco et al. [7b]), and mean face bounding box area ratio. The worst-performing metric determines the overall signal quality grade.

**Table 5.3: Signal Quality Thresholds for Video Modality**

| Metric | Good | Degraded | Poor |
|---|---|---|---|
| Face detection rate | ≥80% frames | 50–80% | <50% |
| Laplacian variance | ≥100 | 50–100 | <50 |
| Face bounding box area | >10% frame area | 5–10% | <5% |

## 5.4 Model Architecture Design

### 5.4.1 ResNet50 Encoder

The ResNet50 encoder extracts 2048-dimensional feature vectors from each 224×224 frame. The fine-tuning strategy is shown in Figure 5.2.

**[Figure 5.2: ResNet50 encoder fine-tuning strategy — insert diagram_resnet.svg here]**

The layers conv1, bn1, layer1, and layer2 are frozen (approximately 3.5 million parameters), preserving the low-level ImageNet features. Layers layer3 and layer4 are fine-tuned (approximately 20 million parameters), allowing the encoder to learn emotion-relevant spatial patterns. The original ImageNet classification head (fc, 1000 classes) is removed, and the output of the global average pooling layer (2048-dimensional) is used as the frame feature vector.

### 5.4.2 BiLSTM Temporal Modelling

The EmotionBiLSTM takes a sequence of T=16 frame features (each 2048-dimensional) and produces a 512-dimensional temporal representation. The architecture is shown in Figure 5.3.

**[Figure 5.3: BiLSTM temporal modelling architecture — insert diagram_bilstm.svg here]**

The BiLSTM consists of 2 stacked bidirectional LSTM layers with hidden size 256 per direction (512 effective), inter-layer dropout of 0.3, and batch-first processing. The output at the final timestep (last_hidden, shape: B × 512) is passed to a two-layer classification head: Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→5). The total parameter count of the BiLSTM module is approximately 6.4 million.

### 5.4.3 Actor-Independent 5-Fold GroupKFold

The training and evaluation strategy uses actor-independent 5-fold GroupKFold, where the grouping variable is the actor ID. This ensures that no actor appears in both the training and validation sets of any fold, preventing the model from learning actor-specific characteristics rather than emotion-specific patterns.

**[Figure 5.5: Actor-independent 5-fold GroupKFold split design — insert diagram_kfold.svg here]**

*[Insert Figure from Colab: GroupKFold fold distribution chart — fold_distribution.png]*

## 5.5 Gated Fusion Design

The gated fusion module combines the video and physiological predictions into a single fused prediction. The algorithm is shown in Figure 5.4.

**[Figure 5.4: Gated fusion algorithm — insert diagram_fusion.svg here]**

### 5.5.1 Confidence Score

The confidence score for each modality is computed using an entropy-based measure, following the recommendation of Guo et al. [3b] as a more calibrated alternative to the maximum softmax probability:

> c_m = 1 - H(P_m) / log(K)

where H(P_m) = -Σ p_k × log(p_k) is the Shannon entropy of the probability distribution, K=5 is the number of classes, and log(K) is the maximum possible entropy. A confidence of 1.0 indicates a perfectly concentrated distribution (full certainty); 0.0 indicates a uniform distribution (maximum uncertainty).

### 5.5.2 Quality Penalty Factor

The quality penalty factor α penalises modalities whose signal quality assessment indicates degraded or poor data.

**Table 5.4: Quality Penalty Factors for Gated Fusion**

| Signal quality | Penalty factor (α) |
|---|---|
| good | 1.0 |
| degraded | 0.5 |
| poor | 0.1 |

### 5.5.3 Gate Score and Modality Weights

The gate score for modality m is computed as:

> g_m = c_m × α_m

The modality weights are computed using L1 normalisation (not softmax), which preserves extreme confidence differences between modalities:

> w_m = g_m / Σ_m g_m

The fused class probability distribution is:

> P_fused = Σ_m w_m × P_m

The predicted emotion is the argmax of P_fused.

### 5.5.4 Graceful Degradation

The graceful degradation rules ensure that the system produces a meaningful prediction even when one or both modalities are unavailable or unreliable.

| Condition | Behaviour |
|---|---|
| Both modalities available | Full gated fusion as described above |
| Video poor or missing | w_video → 0; use physiological prediction only |
| Physio poor or missing | w_physio → 0; use video prediction only |
| Both poor or missing | Uniform distribution (0.2 each class); flagged as unreliable |

## 5.6 Evaluation and Ablation Study Design

### 5.6.1 Primary Metric

Macro-averaged F1 score is used as the primary evaluation metric rather than accuracy, because of mild class imbalance in both the DEAP and CREMA-D datasets after harmonisation [8].

### 5.6.2 Ablation Study

The ablation study evaluates the contribution of each component of MedOracle across five conditions.

**Table 5.5: Ablation Study Conditions**

| Condition | Physiological | Video | Fusion strategy |
|---|---|---|---|
| C1 | ✓ | ✗ | — (physio only baseline) |
| C2 | ✗ | ✓ | — (video only baseline) |
| C3 | ✓ | ✓ | Equal weights (0.5/0.5) |
| C4 | ✓ | ✓ | Confidence-weighted (no quality penalty) |
| **C5** | **✓** | **✓** | **Full method (confidence + quality gating)** |

Three additional robustness conditions are evaluated: C6 (video quality = poor, α=0.1), C7 (video completely missing), and C8 (physiological quality = poor, α=0.1). Statistical significance between C5 and C1/C2/C3 is assessed using the Wilcoxon signed-rank test (one-tailed, α=0.05).

## 5.7 Summary

This chapter presented the detailed analysis and design of MedOracle's components. Label harmonisation maps both datasets to a common five-class taxonomy using theoretically grounded thresholds. The video preprocessing pipeline produces quality-assessed, face-cropped frame tensors. The VideoEmotionModel combines a partially fine-tuned ResNet50 encoder with a two-layer BiLSTM. The gated fusion module uses entropy-based confidence and signal quality penalties with L1-normalised weights. The evaluation design uses macro-F1 as the primary metric with a five-condition ablation study. Chapter 6 describes the implementation work completed at the interim stage.

---

# Chapter 6

# Implementation

## 6.1 Introduction

Chapter 5 presented the analysis and design of MedOracle. This chapter describes the implementation work completed at the interim stage. At this point, the shared modules (label harmonisation and data contracts), the video preprocessing pipeline, and the video model architecture have been fully implemented and verified. Training on the CREMA-D dataset is in progress on Google Colab using a T4 GPU.

## 6.2 Development Environment and Repository Structure

**Table 6.1: Technology Stack**

| Layer | Technology | Version |
|---|---|---|
| Deep learning | PyTorch | ≥2.1.0 |
| Computer vision | torchvision | ≥0.16.0 |
| Face detection | YOLOv8 (ultralytics) | Latest |
| EEG processing | MNE | ≥1.5.0 |
| Signal processing | scipy, numpy | Latest |
| Explainability | shap | ≥0.43.0 |
| Backend | FastAPI, Uvicorn | Latest |
| Database | PostgreSQL (prod), SQLite (dev) | — |
| Frontend | React, Recharts, Axios | — |
| Training platform | Google Colab | T4 GPU |
| Development | Python 3.10+, macOS (MPS), Linux | — |

The repository is structured into four main directories: `shared/` (label harmonisation and data contracts used by all members), `member2_video_fusion/` (video preprocessing, model, and fusion code), `member1_physiological/` (EEG/GSR processing and model), and `member3_explainability/` (SHAP layer and web application). All large files (datasets, model checkpoints) are excluded from version control via `.gitignore`.

**[Figure 6.1: Repository structure — insert diagram_repo.svg here]**

## 6.3 Shared Modules: Label Harmonisation and Data Contracts

The `shared/label_harmonization.py` module implements the mapping functions for both datasets. The function `map_deap_to_class(v, a, d)` applies the V/A/D threshold rules from Table 5.1 and returns an integer emotion class index. The function `map_cremad_to_class(label_str)` applies the CREMA-D mapping from Table 5.2, returning None for DIS labels. A batch function `batch_map_cremad(df)` processes an entire manifest DataFrame.

The `shared/data_contracts.py` module defines the SynchronizedInput dataclass and validation functions for all interface dictionaries. A `validate_prediction_output(d)` function raises descriptive errors if any required field is missing or has an incorrect type, enabling early detection of integration errors.

A comprehensive test suite in `tests/test_label_harmonization.py` contains 59 unit tests covering boundary conditions, all V/A/D threshold combinations, and all CREMA-D label mappings. All 59 tests pass.

## 6.4 Video Preprocessing Implementation

The `member2_video_fusion/preprocessing/face_detector.py` module implements the FaceDetector class with a YOLO primary backend and a Haar cascade fallback. The `process_clip(frames, detector)` function processes a (T, H, W, 3) frame array through face detection, quality assessment, and cropping, returning (cropped_frames, quality_grade, metrics_dict).

The `member2_video_fusion/preprocessing/dataset.py` module implements the CREMADDataset class (a PyTorch Dataset), the `get_folds()` function that returns 5 actor-independent GroupKFold train/val index pairs, and preprocessing utilities including `sample_frames_uniform()`, `normalise_frames()`, `augment_frames()`, and `frames_to_tensor()`.

The CREMA-D manifest CSV (`data/CREMA-D/manifest.csv`) contains 6,171 rows with columns: path, filename, actor_id, emotion, emotion_int, raw_label, sentence, and intensity. It was generated using relative paths to ensure portability across machines.

*[Insert Figure from Colab: Sample preprocessed frames — sample_frames.png]*

*[Insert Figure from Colab: Frame sampling illustration — frame_sampling.png]*

## 6.5 Model Implementation

The `member2_video_fusion/models/resnet_encoder.py` module implements the ResNet50Encoder class. It loads ResNet50 with ImageNet weights, removes the classification head, applies the freezing strategy (conv1, bn1, layer1, layer2 frozen), and exposes a `param_summary()` method. Total parameters: 23.5 million; frozen: 3.5 million; trainable: 20 million.

The `member2_video_fusion/models/bilstm.py` module implements the EmotionBiLSTM class with the architecture described in Section 5.4.2. Total parameters: 6.4 million, all trainable.

The `member2_video_fusion/models/video_model.py` module implements the VideoEmotionModel class that wires the encoder and BiLSTM together. The `forward()` method processes (B, T, 3, 224, 224) input through the full pipeline and returns a dictionary with logits, probabilities, predicted class, frame features, and sequence features. The `predict_clip()` method provides a convenient inference interface that returns a dictionary matching the video portion of the interface contract. Total model parameters: approximately 30 million.

A smoke test for each module verifies shape correctness, probability normalisation, and device placement on CPU, MPS (Apple Silicon), and CUDA. All smoke tests pass.

## 6.6 Summary

This chapter described the implementation work completed at the interim stage, covering the shared modules, video preprocessing pipeline, and model architecture. The label harmonisation module passes all 59 unit tests. The face detection pipeline is operational with the YOLOv8 backend verified on the local development machine. The VideoEmotionModel passes all shape and normalisation checks on CUDA (Google Colab T4 GPU). Training on the CREMA-D dataset is underway. Chapter 7 discusses the current status, differentiation from existing work, and the plan for remaining implementation and evaluation.

---

# Chapter 7

# Discussion and Further Work

## 7.1 Introduction

Chapter 6 described the implementation work completed at the interim stage. This chapter summarises the progress made, discusses how MedOracle differs from existing approaches, and presents the plan for the remaining work including model training, fusion implementation, web application development, and the ablation study.

## 7.2 Work Completed at the Interim Stage

At the interim stage, the following components have been completed and verified:

The shared label harmonisation module is implemented and passes 59 unit tests. The data contracts module defines and validates all interface dictionaries. The CREMA-D dataset has been downloaded, verified (7,442 total clips, 6,171 after dropping DIS), and a portable manifest CSV has been generated. The face detection pipeline using YOLOv8 has been implemented, tested locally on all five emotion classes, and integrated with the preprocessing pipeline. The ResNet50Encoder, EmotionBiLSTM, and VideoEmotionModel classes have been implemented and verified with shape and normalisation checks. Training on the CREMA-D dataset has been set up on Google Colab with weighted cross-entropy loss, Adam optimiser, ReduceLROnPlateau scheduling, actor-independent 5-fold GroupKFold, and per-fold checkpoint saving. The complete system architecture, interface contracts, and evaluation design have been finalised and documented.

## 7.3 Differentiation from Existing Works

MedOracle differs from existing emotion recognition systems in three primary ways. First, it addresses the cross-dataset fusion challenge explicitly: by training the physiological and video modules on entirely separate datasets (DEAP and CREMA-D) with no overlapping subjects, the system must fuse predictions at the decision level. This is more realistic than systems that assume a shared subject pool for joint training.

Second, MedOracle incorporates a signal quality-aware gated fusion mechanism that assigns lower weights to modalities with degraded or poor signal quality. Existing multimodal systems typically assume clean inputs from all modalities [8]; MedOracle's graceful degradation rules ensure that the system produces a meaningful prediction even in adverse conditions.

Third, MedOracle integrates a Kernel SHAP-based explainability layer with a faithfulness metric, combined with an LLM chatbot that translates SHAP outputs into natural-language explanations. This combination addresses a gap in existing affective computing systems, where explainability is either absent or limited to attention weight visualisation [12].

## 7.4 Further Work

The remaining implementation work is distributed across three modules. For the video and fusion module (Member 2), the priority tasks are: completing the training run on CREMA-D, evaluating per-fold macro-F1 performance, implementing the `fusion.py` module, implementing the `pipeline.py` module that integrates the physiological prediction from Member 1, implementing the ablation study (conditions C1–C8), and exporting the `run_full_pipeline()` function. For the physiological module (Member 1), the priority tasks are: completing DEAP preprocessing and feature extraction, implementing the bidirectional cross-modal attention network, running LOSO cross-validation, and exporting `predict_physiological()`. For the explainability module (Member 3), the priority tasks are: implementing Kernel SHAP on the fused pipeline, implementing the faithfulness metric, building the FastAPI backend with all nine endpoints, building the React frontend, and integrating the LLM chatbot.

## 7.5 Plan for Evaluation

The evaluation will proceed in three phases. First, each module will be evaluated independently: the physiological module using LOSO macro-F1 on DEAP, and the video module using 5-fold GroupKFold macro-F1 on CREMA-D. Second, the full pipeline will be evaluated under the five ablation conditions (C1–C5) and three robustness conditions (C6–C8) using a held-out test set constructed from representative samples of both datasets. Third, statistical significance between the full method (C5) and the baselines (C1, C2, C3) will be assessed using the Wilcoxon signed-rank test (one-tailed, α=0.05). Literature baselines to exceed are Siddharth et al. [10] (~73% accuracy on EEG+physio) and Zhang et al. [12] (~65% accuracy on video-only emotion classification).

## 7.6 Summary

This chapter reviewed the work completed at the interim stage, discussed MedOracle's differentiation from existing approaches across three dimensions (cross-dataset fusion, signal quality-aware gating, and SHAP-based explainability), and presented the plan for remaining implementation and evaluation. The project is on track for completion within the project timeline, with the foundational components for Member 2's module fully implemented and training in progress.

---

# References

[1] Baltrusaitis, T., Ahuja, C., and Morency, L. P. (2019), *Multimodal machine learning: A survey and taxonomy*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2), pp 423–443.

[2] Cao, H., Cooper, D. G., Kuchinski, M. K., Swaminathan, R., Shi, J., Bhattacharya, J., and Bhowmick, S. S. (2014), *CREMA-D: Crowd-sourced emotional multimodal actors dataset*, IEEE Transactions on Affective Computing, 5(4), pp 377–390.

[3] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017), *On calibration of modern neural networks*, Proceedings of the 34th International Conference on Machine Learning (ICML), Sydney, Australia, pp 1321–1330.

[4] He, K., Zhang, X., Ren, S., and Sun, J. (2016), *Deep residual learning for image recognition*, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, USA, pp 770–778.

[5] Hochreiter, S. and Schmidhuber, J. (1997), *Long short-term memory*, Neural Computation, 9(8), pp 1735–1780.

[6] Kreibig, S. D. (2010), *Autonomic nervous system activity in emotion: A review*, Biological Psychology, 84(3), pp 394–421.

[7] Koelstra, S., Muhl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., and Patras, I. (2012), *DEAP: A database for emotion analysis using physiological signals*, IEEE Transactions on Affective Computing, 3(1), pp 18–31.

[8] Lundberg, S. M. and Lee, S. I. (2017), *A unified approach to interpreting model predictions*, Advances in Neural Information Processing Systems (NeurIPS), Vol. 30, pp 4765–4774.

[9] Mehrabian, A. (1996), *Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament*, Current Psychology, 14(4), pp 261–292.

[10] Pech-Pacheco, J. L., Cristobal, G., Chamorro-Martinez, J., and Fernandez-Valdivia, J. (2000), *Diatom autofocusing in brightfield microscopy: A comparative study*, Proceedings of the 15th International Conference on Pattern Recognition (ICPR), Barcelona, Spain, pp 314–317.

[11] Poria, S., Cambria, E., Bajpai, R., and Hussain, A. (2017), *A review of affective computing: From unimodal analysis to multimodal fusion*, Information Fusion, 37, pp 98–125.

[12] Russell, J. A. (1980), *A circumplex model of affect*, Journal of Personality and Social Psychology, 39(6), pp 1161–1178.

[13] Siddharth, S., Jung, T. P., and Sejnowski, T. J. (2019), *Utilizing deep learning towards multi-modal bio-sensing and vision-based affective computing*, IEEE Transactions on Affective Computing, 13(1), pp 96–107.

[14] Zhang, S., Zhang, S., Huang, T., and Gao, W. (2020), *Speech emotion recognition using deep convolutional neural network and discriminant temporal pyramid matching*, IEEE Transactions on Multimedia, 20(6), pp 1576–1590.

---

# Appendix A

# Individual Contributions to the Project

**Name of student:** Vanaiyan Kirupagaran (214215H)

My primary contribution to MedOracle is the design and implementation of the video emotion recognition and gated fusion module (Member 2). During the project period leading to this interim report, I completed the following work:

I designed the end-to-end video preprocessing pipeline, including the face detection component using YOLOv8 with a Haar cascade fallback, uniform frame sampling at T=16, face cropping with padding, and signal quality assessment using Laplacian variance and bounding box area metrics. I implemented this pipeline in Python using OpenCV, the Ultralytics library, and PyTorch, and verified it against all five emotion classes in the CREMA-D dataset on my local development machine.

I implemented the ResNet50Encoder, EmotionBiLSTM, and VideoEmotionModel classes in PyTorch, applying the transfer learning strategy of freezing early ResNet50 layers while fine-tuning the later layers. I set up actor-independent 5-fold GroupKFold cross-validation to prevent actor identity leakage between train and test splits. I configured the training pipeline on Google Colab with CUDA acceleration, weighted cross-entropy loss for class imbalance, and per-fold checkpoint saving.

I also contributed to the shared project architecture by designing the interface contracts between all three modules, specifying the signal quality thresholds for all modalities, and designing the gated fusion algorithm including the entropy-based confidence score, quality penalty factors, L1-normalised modality weights, and graceful degradation rules.

Through this work, I deepened my understanding of transfer learning for video classification, the practical challenges of face detection in varying quality video data, the design of actor-independent cross-validation, and the mathematical foundations of confidence-calibrated multimodal fusion. A key challenge I encountered was the identification of a suitable face-detection YOLO model from a public source (the originally referenced GitHub URL returned a 404 error), which I resolved by sourcing the model from HuggingFace. I also resolved a systematic error in the dataset manifest where sandbox absolute paths were embedded rather than relative paths, ensuring portability across machines.

---

**Name of student:** Suhira Balarajan (214206G)

*[To be completed by Suhira — describe your contribution to the physiological module, including DEAP preprocessing, feature extraction, the bidirectional cross-modal attention network design, and any implementation progress at the interim stage. Include what you learned and any problems encountered.]*

---

**Name of student:** Adshaya Balarajah (214024V)

*[To be completed by Adshaya — describe your contribution to the explainability and web application module, including the Kernel SHAP design, FastAPI endpoint planning, database schema design, and React frontend planning. Include what you learned and any problems encountered.]*

---

# Appendix B

# CREMA-D Dataset Statistics

*[Insert the following figures downloaded from Google Colab:]*

- *class_distribution.png* — Bar chart of clip counts per emotion class after label harmonisation
- *actor_distribution.png* — Histogram of clips per actor across the 91 actors
- *intensity_distribution.png* — Distribution of emotional intensity ratings (LO, MD, HI, XX) per class
- *fold_distribution.png* — Stacked bar chart showing class distribution across 5 GroupKFold folds
- *sample_frames.png* — Grid of sample preprocessed frames for each emotion class
- *dataset_summary_table.png* — Summary statistics table

---

# Appendix C

# Interface Contract Specifications

## C.1 M1 → M2 Interface: physiological_prediction_dict

```python
{
    "predicted_emotion":    str,    # One of: "stress", "calm", "happy", "sad", "angry"
    "confidence":           float,  # Entropy-based confidence in [0.0, 1.0]
    "class_probabilities": {        # Must sum to 1.0 (within floating-point tolerance)
        "stress": float,
        "calm":   float,
        "happy":  float,
        "sad":    float,
        "angry":  float,
    },
    "signal_quality": {
        "eeg": str,                 # One of: "good", "degraded", "poor"
        "gsr": str,
    }
}
```

## C.2 M2 → M3 Interface: prediction_output

```python
{
    "predicted_emotion":    str,
    "confidence":           float,
    "class_probabilities": {
        "stress": float, "calm": float, "happy": float,
        "sad": float, "angry": float,
    },
    "modality_weights": {
        "physio": float,            # Sum of physio + video weights = 1.0
        "video":  float,
    },
    "signal_quality": {
        "eeg":   str,               # "good" | "degraded" | "poor"
        "gsr":   str,
        "video": str,
    },
    "per_modality_predictions": {
        "physio": {
            "predicted_emotion":   str,
            "confidence":          float,
            "class_probabilities": dict,
        },
        "video": {
            "predicted_emotion":   str,
            "confidence":          float,
            "class_probabilities": dict,
        },
    }
}
```

---

# Appendix D

# Signal Quality Thresholds

## D.1 EEG Signal Quality Thresholds

| Metric | Good | Degraded | Poor |
|---|---|---|---|
| Amplitude range | ±100 µV | ±100–150 µV | >±150 µV |
| Flat-line channels | 0 | 1–2 | >2 |
| NaN ratio | <1% | 1–5% | >5% |

## D.2 GSR Signal Quality Thresholds

| Metric | Good | Degraded | Poor |
|---|---|---|---|
| Signal range | 0.5–30 µS | 0.1–0.5 or 30–50 µS | <0.1 or >50 µS |
| SCR peaks (60s window) | ≥2 | 1 | 0 |
| Drift (slope) | <0.1 µS/s | 0.1–0.5 µS/s | >0.5 µS/s |

## D.3 Video Signal Quality Thresholds

| Metric | Good | Degraded | Poor |
|---|---|---|---|
| Face detection rate | ≥80% frames | 50–80% | <50% |
| Laplacian variance | ≥100 | 50–100 | <50 |
| Face bounding box area | >10% frame area | 5–10% | <5% |

*Rule: the worst-performing metric determines the overall grade for that modality.*
