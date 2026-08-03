# Chapter 1

## Introduction

### 1.1 Introduction

This report documents MedOracle — an AI-Driven Multimodal Emotional Recognition System with Explainability — developed as the Level 4 Final Year Project of Team MedOracle, Faculty of Information Technology, University of Moratuwa (2026).

Emotion is not communicated through a single channel. A person under stress may show it in a tightened facial expression, in a racing heart rate, in the electrical conductivity of their skin, or in none of these visibly at all while still registering it physiologically. Traditional emotion-recognition systems, however, are almost always built around a single channel of evidence: a camera that reads facial expressions, a microphone that reads vocal tone, or — less commonly outside research settings — a wearable sensor that reads brain or skin activity. Each of these channels is informative in isolation, but each also has blind spots that the others do not share: a face can be deliberately controlled or occluded, a voice can be flat despite genuine distress, and physiological signals can be affected by unrelated factors such as physical exertion or ambient temperature. A system that commits to only one channel inherits that channel's specific failure modes.

MedOracle is built on the premise that these failure modes are largely independent across modalities, and that a system which measures its own confidence in each channel — rather than assuming both channels are always equally trustworthy — can produce a prediction that is more reliable than either channel alone. Concretely, MedOracle fuses two objective, continuously-available signal families: physiological signals (EEG and GSR) captured from wearable sensors, and facial video captured from an ordinary camera. The two are combined into a single prediction across five emotion classes — stress, calm, happy, sad and angry — via a quality- and confidence-aware gated fusion mechanism, described in full in Chapter 5, and every prediction is accompanied by a SHAP-based explanation delivered through a web application, so that the system's reasoning is visible rather than opaque.

The system is the product of a three-person team, each member responsible for one stage of an end-to-end pipeline: physiological signal processing and classification (Member 1), video-based emotion recognition together with the multimodal fusion layer (Member 2), and explainability together with the web application (Member 3). This division of labour mirrors the natural boundaries of the problem — each modality requires substantially different signal processing and modelling expertise — while the interface contracts described in Chapter 5 allow the three modules to be developed independently and composed into a single working system. This report documents the motivation, related work, design, implementation and, most importantly, the honest and rigorously obtained evaluation of that complete system, including results that were not favourable and the diagnostic work undertaken to understand them.

### 1.2 Background and Motivation

Mental health conditions such as stress, anxiety and depression are widespread, and their assessment still relies overwhelmingly on self-report — questionnaires, mood diaries, and periodic clinical interviews — rather than on continuous, objective measurement. Self-report is valuable but has well-known limitations: it is retrospective rather than real-time, it is subject to recall bias and social desirability bias, and it requires the person being assessed to already have enough insight and vocabulary to characterise their own state. Wearable sensors and cameras offer a route to continuous, objective measurement that does not depend on self-report, and affective computing — the branch of AI concerned with recognising and interpreting human emotion — has correspondingly grown as a research area over the last two decades.

That research has, however, progressed largely along separate tracks. Facial-expression recognition, speech-emotion recognition, text-sentiment analysis and physiological emotion recognition have each developed their own benchmark datasets, feature representations and model families, and are typically evaluated and published independently of one another. Multimodal fusion — combining two or more of these channels — has repeatedly been shown in the survey literature to outperform any single modality when the fusion is done carefully [1, 10], precisely because different signals fail under different circumstances. A face can be occluded, poorly lit, or deliberately neutral; physiological signals cannot be consciously suppressed in the same way, but are comparatively more sensitive to motion artefacts, sensor placement, and — as this project's own evaluation shows in detail in Chapter 7 — to variation between individuals.

That last point deserves particular emphasis, because it directly shaped this project's design. Physiological emotion recognition trained on one group of people has repeatedly been observed, in this project's own experiments and in the wider literature, to transfer poorly to a new, previously unseen person — the cross-subject generalisation problem. The DEAP dataset [6], used for this project's physiological module, illustrates why: emotional response to the same stimulus varies considerably from person to person, both in the self-reported valence/arousal/dominance ratings that DEAP records and in the underlying EEG and GSR patterns that accompany them. Facial video, by contrast, was found in this project to generalise comparatively well across unseen individuals once a sufficiently deep visual encoder is used, because the visual grammar of a facial expression — a raised brow, a tightened jaw — is considerably more consistent across people than the electrophysiological signature of the same emotion. This asymmetry between the two modalities' reliability is the central motivation for MedOracle's fusion strategy: rather than assume both modalities are equally trustworthy and fuse them with fixed or learned-but-static weights, the system measures each modality's real-time confidence and signal quality and adapts its trust accordingly, a design informed by calibration research showing that entropy-based confidence remains a useful, model-agnostic reliability signal even when raw softmax outputs are poorly calibrated [3].

The project is further motivated by the need for explainability in any system intended to support mental-health-adjacent decisions. A bare prediction of "stress" carries limited clinical or personal value on its own; a prediction accompanied by a feature-level explanation of which modality drove it, and how reliable that modality's input was judged to be, is considerably more actionable, more trustworthy, and more useful for a person trying to understand their own emotional patterns over time. This is the motivation behind Member 3's SHAP-based explainability layer and the modality-conflict explanation feature described in Chapter 4.

### 1.3 Problem in Brief

Three specific, interrelated problems motivate this project's design and are each addressed directly by a corresponding design decision.

First, existing multimodal emotion-recognition prototypes generally assume every modality is always present and equally reliable, and combine them using fixed or learned-but-static fusion weights that do not adapt at inference time to the quality of the specific input received. A system built this way performs well when all sensors are functioning cleanly, but has no principled way to respond when one input is degraded, noisy, or simply absent — exactly the conditions a real deployment (a poorly-lit camera, a loose EEG electrode) will regularly encounter. MedOracle addresses this by implementing a gated fusion mechanism (Chapter 5) that computes an entropy-based confidence score and a signal-quality grade for each modality at every prediction, multiplies them into a per-modality gate score, and L1-normalises the result into fusion weights — so a degraded or missing modality is automatically down-weighted rather than allowed to corrupt the fused output.

Second, physiological datasets such as DEAP are recorded from a modest number of subjects (32, in DEAP's case) and, once mapped onto a discrete emotion taxonomy, frequently produce a severely imbalanced label distribution; combined with the cross-subject generalisation problem described above, this makes subject-independent evaluation of physiological emotion recognition a genuine and, in much of the published literature, under-reported challenge. Rather than report only an optimistic random-split evaluation — in which windows from the same subject and even the same trial can appear in both the training and test sets — MedOracle's physiological module is evaluated under the harder, honest Leave-One-Subject-Out (LOSO) protocol, and the resulting limitation is diagnosed rather than concealed (Section 7.2).

Third, no single public dataset provides EEG, GSR and facial video recorded from the same subjects at the same time: DEAP provides physiological signals from one group of subjects, and CREMA-D [2] provides facial video from an entirely different group. A system that wants to evaluate genuine multimodal fusion behaviour therefore cannot simply hold out a slice of one paired dataset, and must instead construct a principled cross-dataset evaluation methodology. MedOracle addresses this by pairing, for each of the five emotion classes, a real DEAP physiological window with a real CREMA-D video clip carrying the same emotion label — the standard construction for evaluating cross-dataset late fusion when no single dataset spans every modality [1, 10] — and uses this construction both for the ablation study (Chapter 7) and for a demonstration dataset bundled with the web application.

#### 1.3.1 Scope and Delimitations

The scope of this project is the design, implementation, and rigorous evaluation of a research prototype that accepts pre-recorded EEG, GSR, and facial-video inputs and returns a fused, explainable emotion prediction over five discrete classes. The system is intended as a proof-of-concept for quality-aware multimodal fusion and explainable affective computing, not as a certified clinical diagnostic instrument. Several boundaries follow directly from this scope and from the datasets available.

First, the physiological module is trained and evaluated on DEAP [6], in which subjects viewed music videos in a controlled laboratory setting; the video module is trained on CREMA-D [2], comprising acted emotional utterances from 91 actors. Neither dataset represents real-time, free-living mental-health monitoring, and the cross-dataset pairing used for fusion evaluation (Section 7.4) does not constitute naturally synchronised multimodal recordings from the same person at the same moment. Results reported in this report therefore characterise fusion *behaviour* under a principled constructed evaluation, rather than clinical validity in deployment.

Second, the web application accepts uploaded files (.mp4 video; .npy or .csv EEG/GSR arrays) rather than streaming live sensor data. Real-time acquisition, device driver integration, and latency optimisation are outside the completed scope and are noted as deployment-oriented future work (Section 8.2).

Third, emotion labels are harmonised onto a shared five-class taxonomy (stress, calm, happy, sad, angry) that necessarily simplifies both DEAP's continuous valence/arousal/dominance ratings and CREMA-D's six original categories. Any discrete mapping of dimensional affect onto categorical labels is itself a modelling decision with direct consequences for class balance and evaluability, as the physiological module's diagnostic experiments demonstrate (Section 7.2).

Fourth, explainability is provided through SHAP-based feature attribution and natural-language summaries generated by a large language model. These outputs are intended to help a user understand *which modality and features influenced a given prediction*, not to provide clinical advice, and their usefulness to end users has not yet been validated through a formal user study (Section 7.7).

Within these boundaries, the project delivers a complete, testable pipeline — from raw modality inputs through gated fusion to a JWT-secured web dashboard — together with an eight-condition ablation study and an honest account of both favourable and unfavourable findings.

### 1.4 Aim and Objectives

#### 1.4.1 Aim

To design, implement and rigorously evaluate a multimodal emotion recognition system that fuses physiological (EEG/GSR) and facial-video signals through a quality-aware gated fusion mechanism, and to make its predictions explainable through SHAP-based feature attribution delivered via a web application.

#### 1.4.2 Objectives

In pursuit of this aim, the project set out to achieve the following specific objectives, each of which is revisited and evaluated against its outcome in Chapters 7 and 8:

- To develop a physiological emotion recognition module using a bidirectional cross-modal attention network trained on the DEAP dataset, mapped to a five-class emotion taxonomy shared across the system.
- To develop a video-based emotion recognition module using face detection, convolutional feature extraction and temporal (BiLSTM) modelling, trained on the CREMA-D dataset, and evaluated under an actor-independent cross-validation protocol so that reported performance reflects genuine generalisation to unseen individuals.
- To design and implement a quality-aware gated fusion mechanism that combines both modalities using entropy-based confidence and signal-quality penalties, with graceful degradation when a modality is degraded or missing, and to verify this mechanism's correctness through automated unit testing.
- To implement a SHAP-based explainability layer and a FastAPI/React web application that allows a user to submit video and/or physiological data and receive a fused, explained prediction, including an explanation of disagreements between modalities.
- To evaluate the complete system through an eight-condition ablation study (single-modality baselines, naive fusion, the full method, and robustness conditions under degraded or missing input) with statistical significance testing, and to report all resulting findings — including unfavourable ones — with full transparency about their cause.

#### 1.4.3 Research Questions

The problems stated in Section 1.3 give rise to four research questions that this report addresses empirically:

**RQ1.** Can physiological (EEG/GSR) and facial-video signals, trained on separate public datasets with no shared subjects, be combined through late fusion to produce a prediction that significantly outperforms either modality in isolation?

**RQ2.** Does a quality- and confidence-aware gated fusion mechanism degrade gracefully — down-weighting or excluding a modality when its input is poor or absent — without corrupting the fused output or causing system failure?

**RQ3.** Under subject-independent evaluation protocols (Leave-One-Subject-Out for physiology; actor-independent cross-validation for video), what performance can each unimodal branch achieve, and what diagnostic factors explain any gap between optimistic and honest evaluation figures?

**RQ4.** Can SHAP-based attribution, together with a web application that surfaces modality weights, signal-quality grades, and modality-conflict explanations, make the fused prediction interpretable to a non-technical user?

RQ1 and RQ2 are answered primarily in Section 7.4 (ablation study). RQ3 is answered in Sections 7.2 and 7.3. RQ4 is addressed in Sections 6.4 and 7.5, with limitations noted in Section 7.7.

### 1.5 Proposed Solution

The proposed system ingests three raw input types — EEG, GSR and facial video — and produces a single fused emotion label with an accompanying confidence score, a per-modality prediction breakdown, and a SHAP-based explanation. Figure 1.1 shows the high-level architecture: Member 1's physiological branch and Member 2's video branch each independently produce a prediction dictionary conforming to a shared interface contract; Member 2's gated fusion module combines these into a single `prediction_output`; and Member 3's explainability layer and web application surface the result, together with its explanation, to the end user. Table 1.1 summarises the three modules and their ownership before each is described in more detail in Sections 1.5.1–1.5.3.

![Figure 1.1: High-level architecture of the proposed multimodal emotion recognition system](../report_assets/fig_architecture.png)

**Figure 1.1: High-level architecture of the proposed multimodal emotion recognition system**

**Table 1.1: Summary of system modules and responsibilities**

| Module | Owner | Responsibility |
|---|---|---|
| Physiological Emotion Recognition | Member 1 (Suhira Balarajan) | EEG+GSR preprocessing, bidirectional cross-modal attention network, LOSO evaluation |
| Video Emotion Recognition + Fusion | Member 2 (Vanaiyan Kirupagaran) | Face detection, ResNet50+BiLSTM video model, gated fusion, full pipeline, ablation study |
| Explainability + Web Application | Member 3 (Adshaya Balarajah) | Kernel SHAP, FastAPI backend, React dashboard, LLM-based chat explanations |

**Table 1.2: Shared five-class emotion taxonomy (locked across all modules)**

| Class index | Label | Description (operational) |
|---|---|---|
| 0 | stress | High arousal, low valence, low dominance (DEAP mapping); fear mapped from CREMA-D |
| 1 | calm | Low arousal, neutral-to-positive valence; CREMA-D neutral mapped here |
| 2 | happy | High valence, high arousal, high dominance |
| 3 | sad | Low valence, low arousal, low dominance |
| 4 | angry | Low valence, high arousal, high dominance |

The full DEAP and CREMA-D mapping rules, including dropped labels and threshold conventions, are specified in Section 5.3 and Table 5.2.

#### 1.5.1 Member 1 — Physiological Emotion Recognition Module

Member 1's module takes raw EEG (32 channels, 128Hz) and GSR signals recorded during the DEAP dataset's music-video-viewing sessions, and produces an emotion prediction over the shared five-class taxonomy. Each 60-second DEAP trial is windowed into 4-second segments, and each trial's continuous self-reported valence, arousal and dominance ratings are mapped onto the shared five-class taxonomy using literature-grounded thresholds (Russell's circumplex model [11] and Mehrabian's PAD model [8], detailed in Chapter 5). The EEG and GSR windows are each passed through a dedicated convolutional encoder, after which a bidirectional cross-modal attention layer allows the EEG representation to attend over the GSR representation and vice versa — an architecture intended to let each signal inform the interpretation of the other, rather than treating them as independent evidence to be combined only at the very end. The attended representations are pooled and passed to a joint classification head producing a five-class probability distribution. Critically, this module is evaluated twice: once under an optimistic random-split protocol used only to obtain an early working checkpoint for system integration, and once under the honest, subject-independent Leave-One-Subject-Out protocol that this report treats as the module's true, reportable performance figure. The substantial gap between these two evaluations, and the diagnostic work undertaken to explain it, is one of this project's central findings and is discussed at length in Section 7.2.

#### 1.5.2 Member 2 — Video-Based Emotion Recognition and Gated Fusion Module

Member 2's module has two parts. The first is a video-based emotion recognition pipeline: sixteen frames are sampled uniformly from each input clip and passed through a partially fine-tuned ResNet50 [4] convolutional encoder to obtain per-frame feature vectors, which a two-layer bidirectional LSTM [5] then models temporally before a final softmax layer produces a five-class prediction; a YOLOv8 face-detection pass runs alongside this pipeline purely to grade the input's signal quality (face-detection rate, sharpness, face size) rather than to crop the recognition input itself, a design choice justified by a controlled comparison described in Chapter 6. The second part is the gated fusion mechanism that combines this video prediction with Member 1's physiological prediction: an entropy-based confidence score and a signal-quality grade are computed for each modality, multiplied together into a gate score, and L1-normalised into fusion weights, so that the fused prediction automatically leans on whichever modality is currently more reliable, and degrades gracefully — falling back to a single modality's prediction, or to a flagged uniform distribution — when a modality is degraded or unavailable. Member 2 additionally designed the full multimodal pipeline that invokes both members' models, the cross-dataset evaluation-set construction needed because DEAP and CREMA-D share no subjects, and the eight-condition ablation study with statistical significance testing that forms the core of this report's evaluation (Chapter 7).

#### 1.5.3 Member 3 — Explainability and Web Application Module

Member 3's module receives the fused `prediction_output` produced by Member 2's pipeline and makes it interpretable and accessible. A Kernel SHAP explainability layer attributes the fused prediction to each contributing modality and feature, and computes a faithfulness score by perturbing input features and measuring the resulting change in prediction, so that the explanation's accuracy can itself be quantified rather than assumed. A FastAPI backend, secured with JWT authentication, exposes this functionality through a set of endpoints covering user registration and login, video-only and full multimodal (video + EEG + GSR) prediction, SHAP explanation retrieval, session history, and an LLM-backed chat interface that translates the raw SHAP attribution into a plain-English explanation a non-technical user can act on. A React frontend renders this information as an emotion-trend chart, a SHAP bar chart, a session-history panel, and — Member 3's novel contribution — a modality-conflict explanation panel that specifically explains cases where the physiological and video predictions disagree, surfacing the same reasoning the gated fusion mechanism used internally to decide which modality to trust.

### 1.6 Report Organisation

The remainder of this report is structured as follows. Chapter 2 reviews prior work in physiological, video-based, and multimodal emotion recognition. Chapter 3 describes the technologies adopted. Chapter 4 describes each member's approach in terms of input, process, output, and evaluation. Chapter 5 presents the system's analysis and design, including the shared label taxonomy, interface contracts, and fusion design. Chapter 6 describes the implementation. Chapter 7 presents the evaluation and discussion, including the ablation study and an honest account of the physiological module's limitations. Chapter 8 concludes the report, maps each objective to its outcome, and outlines future work. Appendix A records each team member's individual contribution.

### 1.7 Summary

This chapter introduced MedOracle, a multimodal emotion recognition system that fuses physiological and video signals through a quality-aware gated fusion mechanism and explains its predictions through SHAP. The motivation for combining modalities, three interrelated problems addressed by the design, the project's scope and delimitations, aim, objectives, and research questions, and an overview of each member's module within the proposed solution were presented. The next chapter reviews relevant prior work in each of the constituent research areas.
