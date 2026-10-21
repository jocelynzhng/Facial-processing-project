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
