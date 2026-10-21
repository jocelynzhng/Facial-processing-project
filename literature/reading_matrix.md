# Literature Reading Matrix

## 1. DeepFace — Taigman et al. (2014)

**Paper:** *DeepFace: Closing the Gap to Human-Level Performance in Face Verification*

| Field | Summary |
|---|---|
| **Research Question** | Can a deep neural network learn a face representation that makes face verification robust to real-world variations such as pose and lighting? |
| **Data** | Training: ~4 million facial images from 4,000+ identities. Evaluation includes Labeled Faces in the Wild (LFW). |
| **Target** | Face identity / verification: determine whether two face images represent the same person. |
| **Input Representation** | Detected faces → 3D face alignment → aligned/cropped facial images. |
| **Method** | 9-layer deep neural network using convolutional layers, locally connected layers, and fully connected layers. |
| **Validation** | Evaluated using established face-verification protocols such as LFW. Benchmark identities are separate from the identities used for training. |
| **Evaluation** | Primarily face-verification accuracy. |
| **Main Result** | Achieved 97.35% accuracy on LFW, approaching human-level performance at the time. |
| **Limitations** | Requires a very large proprietary training dataset; computationally expensive for its era; relies heavily on explicit 3D alignment; potential demographic bias in training data. |
| **Relevance to Project** | Important historical baseline showing the transition from handcrafted facial features to learned deep representations. Introduces the general pipeline: detect → align → represent → verify. |

**Key takeaway:** DeepFace showed that deep neural networks could learn highly effective facial representations, especially when combined with careful face alignment.

---

## 2. FaceNet — Schroff et al. (2015)

**Paper:** *FaceNet: A Unified Embedding for Face Recognition and Clustering*

| Field | Summary |
|---|---|
| **Research Question** | Can faces be mapped directly into an embedding space where distance represents facial identity similarity? |
| **Data** | Very large private face dataset for training. Evaluation includes LFW and YouTube Faces (YTF). |
| **Target** | Face identity similarity rather than simply predicting an identity class. |
| **Input Representation** | Roughly aligned face crops. Training examples are organised as anchor-positive-negative triplets. |
| **Method** | Deep CNN + triplet loss + online triplet mining. Learns a compact Euclidean face embedding. |
| **Validation** | Uses established verification protocols on identities not used for training. Triplets are constructed during training rather than using a simple random image-level split. |
| **Evaluation** | Face-verification accuracy and recognition/clustering performance. |
| **Main Result** | Achieved 99.63% accuracy on LFW and 95.12% on YouTube Faces. |
| **Limitations** | Triplet selection/mining can make training difficult; requires large amounts of identity-labelled face data; results depend on the composition of the training dataset. |
| **Relevance to Project** | Introduces metric learning. Useful for understanding similarity, clustering, and embedding-based approaches rather than simple classification. |

**Key takeaway:** FaceNet shifted the problem from "Which identity is this?" to "How close are these two faces in a learned representation space?"

---

## 3. ArcFace — Deng et al. (2019)

**Paper:** *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*

| Field | Summary |
|---|---|
| **Research Question** | How can the loss function make face embeddings more compact within identities and more separated between different identities? |
| **Data** | Large-scale face-recognition datasets with evaluation across 10 established face-recognition benchmarks, covering different challenges, such as image quality, pose variations, and video recognition. |
| **Target** | Face identity / verification / recognition. |
| **Input Representation** | Aligned facial images passed through a deep CNN to produce normalized embedding vectors. |
| **Method** | Deep CNN + Additive Angular Margin Loss (ArcFace). Faces are represented on a normalized hypersphere and identities are separated using angular margins. |
| **Validation** | Uses standard benchmark-specific verification and identification protocols. |
| **Evaluation** | Verification accuracy, identification rates, TAR/FAR, and other benchmark-specific recognition metrics. |
| **Main Result** | Improved performance over previous margin-based face-recognition methods across numerous benchmarks. |
| **Limitations** | Requires large identity-labelled datasets; primarily designed for identity discrimination rather than other facial properties such as emotion or perception; potential dataset bias. |
| **Relevance to Project** | Demonstrates how changing the loss function can significantly improve the quality and geometry of learned visual representations. |

**Key takeaway:** ArcFace improved face embeddings by explicitly increasing angular separation between identities.

---
## 4. Vision Transformer (ViT) — Dosovitskiy et al. (2021)

**Paper:** *An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale*

| Field | Summary |
|---|---|
| **Research Question** | Can a Transformer, without a CNN backbone, effectively learn visual representations from images? |
| **Data** | Large labelled datasets including ImageNet-21k and JFT-300M; evaluated on ImageNet, CIFAR-10/100, VTAB, and other datasets. |
| **Target** | Primarily general image classification, not facial identity specifically. |
| **Input Representation** | Image → divided into fixed-size patches (e.g., 16×16) → flattened → converted into patch embeddings → positional information added. |
| **Method** | Vision Transformer (ViT) using multi-head self-attention instead of convolution as the primary image-processing mechanism. |
| **Validation** | Pretrained on large datasets and evaluated/fine-tuned on separate established datasets. |
| **Evaluation** | Primarily top-1 image-classification accuracy. |
| **Main Result** | Demonstrated that pure Transformers can match or outperform strong CNNs when pretrained on sufficiently large datasets. |
| **Limitations** | Data hungry; requires substantial pretraining; large models are computationally expensive. |
| **Relevance to Project** | Introduces the major architecture transition from CNNs to Transformers and provides the architectural basis for later models such as DINO. |

**Key takeaway:** ViT demonstrated that images can be processed as sequences of patches using Transformer attention rather than convolution.

---
## 5. DINO — Caron et al. (2021)

**Paper:** *Emerging Properties in Self-Supervised Vision Transformers*

| Field | Summary |
|---|---|
| **Research Question** | What visual representations emerge when a Vision Transformer is trained without human-provided labels? |
| **Data** | Primarily ImageNet images; labels are not used during DINO pretraining. |
| **Target** | No conventional target label during pretraining. The objective is to learn general visual representations. |
| **Input Representation** | Multiple augmented/cropped views of the same image, represented as ViT image patches. |
| **Method** | Self-distillation with no labels (DINO). Teacher and student networks receive different views of an image and are encouraged to produce consistent representations. |
| **Validation** | Learned representations are evaluated using methods such as k-nearest-neighbour classification and linear evaluation. |
| **Evaluation** | ImageNet k-NN accuracy, linear-probe accuracy, and analysis of learned attention/semantic structure. |
| **Main Result** | A small self-supervised ViT achieved 78.3% ImageNet k-NN top-1 accuracy, while ViT-Base achieved 80.1% top-1 under linear evaluation. Attention maps also revealed meaningful semantic object structure without segmentation labels. |
| **Limitations** | Significant computational requirements; performance depends on augmentation and teacher/student training design; not specifically designed for faces. |
| **Relevance to Project** | Highly relevant to visual perception because it demonstrates that meaningful visual structure can emerge without explicit labels. Useful when working with relatively small labelled datasets. |

**Key takeaway:** DINO showed that self-supervised ViTs can learn meaningful visual and spatial representations without manually labelled training data.

---
## 6. AdaFace — Kim, Jain & Liu (2022)

**Paper:** *AdaFace: Quality Adaptive Margin for Face Recognition*

| Field | Summary |
|---|---|
| **Research Question** | Should difficult face images be treated differently depending on whether they are informative hard examples or simply low-quality/unrecognisable images? |
| **Data** | Face-recognition training datasets; evaluation includes challenging benchmarks such as IJB-B, IJB-C, IJB-S, and TinyFace. |
| **Target** | Face identity / recognition under varying image quality. |
| **Input Representation** | Aligned/cropped facial images. Image quality is indirectly estimated using the norm of the network's feature vector. |
| **Method** | Deep face-recognition network + quality-adaptive margin loss. Margin behaviour changes according to estimated image quality. |
| **Validation** | Uses standard person-based face-recognition benchmark protocols rather than random image-level splitting. |
| **Evaluation** | Verification and identification metrics across challenging face-recognition benchmarks. |
| **Main Result** | Improved recognition over previous margin-loss approaches on several challenging low-quality face benchmarks. |
| **Limitations** | Image quality is approximated through feature magnitude rather than directly measured; designed primarily for identity recognition. |
| **Relevance to Project** | Relevant if facial images vary in blur, resolution, lighting, occlusion, pose, or camera conditions. Demonstrates that "hard example" and "low-quality example" should not necessarily be treated the same way. |

**Key takeaway:** AdaFace makes face-recognition training quality-aware by adapting the learning margin according to estimated image quality.

---

## 7. DINOv2 — Oquab et al. (2023)

**Paper:** *DINOv2: Learning Robust Visual Features without Supervision*

| Field | Summary |
|---|---|
| **Research Question** | Can large-scale self-supervised learning produce a general-purpose visual representation that transfers effectively to many different tasks? |
| **Data** | Approximately 142 million curated unlabeled images (LVD-142M). |
| **Target** | No explicit class target during pretraining. The objective is to learn useful and transferable visual representations. |
| **Input Representation** | General natural images represented as ViT patches, with multiple augmented views used during self-supervised training. |
| **Method** | Large-scale self-supervised Vision Transformer training incorporating self-distillation and related representation-learning techniques. |
| **Validation** | Pretrained representations are evaluated across numerous independent downstream datasets and tasks. |
| **Evaluation** | Multiple benchmarks covering classification, instance retrieval, semantic segmentation, depth estimation, and other visual tasks. |
| **Main Result** | Produced strong general-purpose visual features that transfer effectively across different tasks and domains, often without full model fine-tuning. |
| **Limitations** | Enormous training-data and compute requirements; unrealistic to train from scratch for a small research project; web-scale data may contain sampling and representation biases. |
| **Relevance to Project** | Highly practical as a pretrained feature extractor. Facial images could be passed through DINOv2 to obtain embeddings, followed by a relatively small downstream classifier or regressor. |

**Key takeaway:** DINOv2 moves toward a general visual foundation model whose pretrained representations can be reused for many downstream tasks.

---
## 8. DINOv3 — Siméoni et al. (2025)

**Paper:** *DINOv3*

| Field | Summary |
|---|---|
| **Research Question** | How far can self-supervised visual representation learning be scaled while maintaining strong dense spatial features useful for detailed visual tasks? |
| **Data** | Large-scale unlabeled image data, scaling to approximately 1.7 billion images. |
| **Target** | No conventional labels during core self-supervised pretraining; the objective is to learn general-purpose image and dense spatial representations. |
| **Input Representation** | General images transformed into Vision Transformer patch tokens. |
| **Method** | Large-scale self-supervised Vision Transformer training. Introduces techniques including Gram anchoring to maintain high-quality dense features during long training. |
| **Validation** | Representations are evaluated across numerous independent downstream vision tasks and datasets. |
| **Evaluation** | Classification and dense prediction benchmarks, including segmentation, detection, and other spatial visual tasks. |
| **Main Result** | Produces very strong general-purpose and dense visual representations and extends self-supervised visual foundation models to substantially larger scales. |
| **Limitations** | Extreme data and computational requirements for original training; not face-specific; large foundation models make it difficult to attribute improvements to a single methodological component. |
| **Relevance to Project** | Particularly relevant if the project eventually investigates where information occurs within a face, such as facial regions, landmarks, local changes, segmentation, or other spatial representations. |

**Key takeaway:** DINOv3 extends self-supervised visual representation learning toward very large foundation models with particularly strong dense/spatial representations.

---

# Methodological Progression

| Stage | Paper | Main Change | Key Question |
|---|---|---|---|
| 1. Deep face learning | DeepFace | Handcrafted → deep facial features | Can deep learning learn identity-relevant facial information? |
| 2. Metric learning | FaceNet | Classification → embedding | How should facial similarity be represented? |
| 3. Better embedding geometry | ArcFace | Triplet/distance → angular margin | How can representations become more separable? |
| 4. New architecture | ViT | CNN → self-attention | Does visual processing require convolution? |
| 5. Self-supervision | DINO | Labelled → unlabeled learning | Can useful visual structure emerge without labels? |
| 6. Image quality | AdaFace | Fixed → quality-adaptive learning | How should low-quality facial images affect training? |
| 7. Foundation features | DINOv2 | Task-specific → general representation | Can one pretrained representation support many visual tasks? |
| 8. Large-scale dense features | DINOv3 | General → strong dense/spatial representation | Can self-supervised learning capture detailed spatial visual information? |

# Overall Synthesis

The literature shows a progression from face-specific supervised deep learning toward general-purpose self-supervised visual representation learning.

**DeepFace → FaceNet → ArcFace/AdaFace**
focus primarily on face identity and improve how facial representations are learned.

**ViT**
changes the underlying architecture from convolution-based image processing to Transformer-based attention.

**DINO → DINOv2 → DINOv3**
move toward self-supervised, general-purpose visual representations that can potentially be transferred to facial and perceptual tasks without training a large model from scratch.

For this project, this suggests that the research direction could be **Facial/Image Processing + Visual Perception + Deep Learning**. Possible downstream outcomes could include:
- facial identity
- facial similarity
- emotion/expression
- expression intensity
- facial landmarks or regions
- perceived facial traits
- image quality
- other perception-related outcomes

A practical future pipeline could therefore be:

**Facial images → pretrained visual model (e.g., DINOv2) → embeddings/features → smaller task-specific classifier or regressor**