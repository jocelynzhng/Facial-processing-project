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