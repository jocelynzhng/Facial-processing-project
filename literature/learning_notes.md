
# Learning Notes on Literature Review

## 1. DeepFace — Taigman et al. (2014)

**Paper:** [DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://openaccess.thecvf.com/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)

### 1.1 Convolutional Layers vs. Locally Connected Layers

- Both can learn facial features.
- **The main difference** is whether the same learned filter is reused at different locations.
  - **Convolutional layers:** The same filter is applied across different locations (**weight sharing**).
  - **Locally connected layers:** Different locations can learn different filters (**no weight sharing**).
- **Early convolutional layers** are good at detecting general, low-level patterns, such as edges, curves, and textures.
  - If an edge detector is useful in one location, it may also be useful elsewhere, so the same filter can be reused.
  - This is called **weight sharing**.
- Pattern detection is a **learned ability**. The network is not explicitly programmed to look for predefined patterns such as curves; it learns useful filters during training.
- **Locally connected layers** can learn location-specific features.
  - For example, because aligned faces usually have noses at roughly the same location, filters operating in that region can specialize in patterns useful there.
  - However, individual filters are not necessarily explicit "nose detectors" or "eye detectors".

**Key distinction:**

> Convolutional layers = local + shared filters  
> Locally connected layers = local + location-specific filters

### 1.2 Overall DeepFace Workflow

```text
Aligned face image
        ↓
Convolutional layers
        ↓
Locally connected layers
        ↓
Fully connected layer (F7)
        ↓
Face representation (4096-dimensional vector)
        ↓
F8 + Softmax (during training)
        ↓
Predicted training identity
```

*Note: This is a simplified workflow showing the main concepts, rather than every layer in the original architecture.*

### 1.3 Face Alignment Explained

Suppose we have three photographs of the same person:

- One facing straight ahead
- One with the head tilted
- One with the head turned

The eyes, nose, and mouth may appear at different pixel coordinates because of changes in head position and camera angle.

**Face alignment** identifies corresponding facial landmarks and transforms the face so that these features appear in approximately **standardized positions**.

This is somewhat analogous to geometrically correcting images in GIS.

**DeepFace's automated alignment pipeline (simplified):**

```text
Raw photograph
       ↓
Detect the face
       ↓
Detect facial landmarks
(e.g., eyes, nose, mouth positions)
       ↓
Fit a 3D face model
       ↓
Transform / frontalize the face
       ↓
Aligned face image
       ↓
Deep Neural Network (DNN)
```

### 1.4 Logic Behind the Order of Layers

- **Early convolutional layers:** "Find useful visual patterns wherever they occur."
- **Later locally connected layers:** "Now that I know where on the aligned face I'm looking, learn features particularly useful for this region."
- **Fully connected layers:** "Combine information from across the face into an overall representation."

### 1.5 Fully Connected Layer (F7)

A **fully connected layer (FCL)** receives information from all outputs of the preceding layer.

It is useful toward the end of DeepFace because the model has already extracted many local features and now needs to combine information across the whole face.

For example, identity is not determined by one isolated feature, but by combinations of many features and their relationships.

**F7:**

- F7 is a fully connected layer.
- For each input face, its output is a **4096-dimensional feature vector**.
- This vector is a learned **face representation**.
- The numbers encode **identity-relevant facial information** learned by the network.
- The representation does not inherently mean "Alice" or identify a particular person by name.

> **F7 output = a reusable numerical representation of a face, rather than a fixed identity label.**

### 1.6 F8: Identity Classification

*Note: F7 and F8 are layer names specific to the DeepFace architecture, not general deep-learning terminology.*

**F8:**

- Receives the learned representation from F7.
- Maps the 4096-dimensional feature vector to scores for the **4030 training identities**.
- Answers the question: "Which of the 4030 people I was trained to classify does this face most likely belong to?"
- Produces a vector of 4030 class scores.

**Relationship between F7 and F8:**

- The network learns facial features while also learning to distinguish the training identities.
- The features learned by F7 can generalize to previously unseen people.
- Researchers can therefore reuse F7 representations for **face verification**.
- F8 is tied to the training identities, while F7 produces a more generalizable representation.

**Key distinction:**

> F7 = What identity-relevant features describe this face?  
> F8 = Which training identity does this representation correspond to?

### 1.7 Summary of Layer Types

| Layer | Main Function | Simple Explanation |
|---|---|---|
| Convolutional | Extract general local patterns | Local + shared rules |
| Locally connected | Extract location-specific patterns | Local + location-specific rules |
| Fully connected (F7) | Combine information across the face | Global combination |
| Fully connected (F8) | Produce training-identity scores | Identity classification |

### 1.8 Softmax

**Softmax** is a function that converts a set of scores into a probability distribution.

Suppose F8 produces the following scores:

```text
Alice:    4.8
Bob:      1.2
Charlie: -0.5
```

Softmax converts them into approximately:

```text
Alice:    96.9%
Bob:       2.6%
Charlie:   0.5%
          ------
          100%
```

The network can then compare its predicted probability distribution with the actual training identity to calculate the loss.

**Important distinction:**

- **Softmax:** Converts class scores into probabilities.
- **Loss function:** Measures prediction error.
- **Optimizer:** Updates network weights to reduce the loss.

Softmax itself is **not an optimizer**.

---

## 1.9. Big Picture: Representation Learning

### 1.9.1 What Is a Learned Representation?

One of the central ideas connecting the papers in this literature review is:

> **How can a neural network learn a useful numerical representation of an image?**

In DeepFace, this representation is the **4096-dimensional feature vector** produced by F7.

A general workflow is:

```text
Raw image
    ↓
Neural network
    ↓
Learned feature representation / embedding
    ↓
Prediction, comparison, or classification
```

The terms **feature representation**, **feature vector**, and **embedding** are closely related, although their precise meanings depend on context.


### 1.9.2 DeepFace → FaceNet

**DeepFace essentially asks:**

> Can a neural network learn facial representations by training it to distinguish among known identities?

**FaceNet instead asks:**

> Rather than learning a representation indirectly through identity classification, can we directly train the representation so that images of the same person lie close together, while images of different people lie farther apart?

This introduces **triplet loss**, which is central to FaceNet's methodology.

**Key methodological shift:**

> DeepFace: Identity classification → learned representation  
> FaceNet: Metric learning → embedding space organized by identity similarity

### 1.9.3 Main Takeaway

The key question connecting these papers is:

**"How do we produce a useful feature representation?"**

Different methods answer this question by changing:

- The **network architecture** (e.g., CNN vs. Transformer).
- The **training objective or loss function** (e.g., classification loss vs. triplet loss).
- The **learning paradigm** (e.g., supervised vs. self-supervised learning).
- The **training data and scale**.
- How the resulting representations are used for downstream tasks.

Understanding these differences is more important at this stage than memorizing the exact architecture of each model.


## 2. FaceNet — Schroff et al. (2015)

**Paper:** [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://openaccess.thecvf.com/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)

### 2.1 Questions to Answer

1. What an embedding is;
2. Why distance between embeddings can represent similarity;
3. Triplet loss and anchor/positive/negative;
4. Difference between classification vs metric learning

### 2.2 Embedding

**Embedding**: a numerical representation (a list of numbers) that a neural network (NN) learns to describe important features of an inpt, such as a face.

- E.g. FaceNet converts a face image into an embedding like [0.21, -0.53, 0.87, ...]. Images of the same person have similar embeddings, and images of different people have more distant embeddings.

### 2.3 Research Question and Embedding Space

**Research question:** Can faces be mapped directly into an **embedding space** where distance corresponds to facial identity similarity?

- Mapping here means to use a NN to convert a face image into numbers that determine where the point sits on the embedding space, which is analogous to points' distances on an Euclidean space.

### 2.4 Evaluation with LFW

**Evaluation with LFW:** LFW evaluates face verification using 6,000 predefined pairs of face images: 3,000 pairs of the same person and 3,000 pairs of different people.

- The general evaluation procedure is standardized: researchers test their models on the same 6,000 face pairs using the same 10 fold evaluation protocol;
- FaceNet converts each image into a numerical embedding, then calculates the Euclidean distance between any two embeddings. If the distance is below a chosen threshold, it predicts "same person"; otherwise, "different people." *This approach where the model learns to represent images so that similar ones are close together and different ones are farther apart in embedding space, is called* **metric learning**.
- **Representation learning:** A broad approach where a model learns useful features (embeddings) from data [For future papers]. *Metric learning* is one approach to representation learning. DeepFace learns representations through classification, while FaceNet learns them directly through similarity relationships.

### 2.5 Youtube Faces (YTF)

**Youtube Faces (YTF):** a face verification dataset containing 3425 video clips of 1595 people.

- Compared to images, videos introduce additional challenges such as movement, changing facial angles, blur and lighting.

### 2.6 DeepFace vs. FaceNet in Their Training Objectives

Both have the ultimate task of face verification.

- **DeepFace - identity classification:** Trains the network to classify a face among known identities, which also learns facial embeddings (indirect). Then, the learned facial embeddings are used for face verification.
- **FaceNet - distance based metric learning:** Trains the network directly to produce embeddings where same person images are close together, using **triplet loss**.

### 2.7 Triplet as Input Representation

- **Anchor:** the reference photo (e.g., Alice's 1st photo)
- **Positive:** Another image of the *same* person (e.g., Alice's 2nd photo)
- **Negative:** An image of a different person (e.g., Bob's photo)

### 2.8 Triplet Loss

**Triplet loss:** a loss function that measures how well a neural network separates images of different people while keeping images of the same person close together in embedding space. For example, if the negative is not sufficiently far away from the anchor than the positive, *triplet loss penalizes the model, encouraging it to adjust its embeddings*.

- In other words, the goal is to make the anchor-negative distance larger than the anchor-positive distance by at least a specified margin.

### 2.9 Online Triplet Mining

**Online triplet mining:** Online triplet mining is the process of selecting useful anchor–positive–negative triplets during model training; It focuses on challenging examples where the model struggles to distinguish identities, since easy triplets contribute little to further learning.

- This automatically happens during training, but researchers must first program the rules for selecting triplets.
- **Offline (triplets prepared beforehand):** Researchers or an algorithm select triplets before training, using a fixed selection or periodically updated selections.
- **Online:** During training, an algorithm automatically examines the model's current embeddings and selects challenging triplets, usually from the current batch of images.
- *The key advantage:* As the model improves, online triplet mining can adapt which triplets are considered difficult, rather than relying on a fixed set.

### 2.10 FaceNet Evaluates 3 Types of Performances

- **Face verification (accuracy):** Given 2 photos, does the model correctly determine if they are the same person; accuracy measures the percentage of correct decisions;
- **Face recognition:** Given a face, can the model *identify* the person by comparing it with *known* identities;
- **Face clustering:** Without being told their identities, can the model automatically group photos of the same person together.

## 3. ArcFace — Deng et al. (2019)

**Paper:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)

### 3.1 Loss Function

**Loss function:** measures how far a NN's predictions or learned representations are from the desired result. During *training*, the model adjusts its weights to minimize the loss, and encourages the model to learn:

- **DeepFace (Softmax cross-entropy loss):** to correctly classify a face among *known* identities;
- **FaceNet (Triplet loss):** to keep embeddings of the same person close together and different people farther apart;
- **ArcFace (Additive angular margin loss):** classify identities correctly while enforcing greater angular separation between different identities in embedding space;
- ArcFace returns to a classification-based training approach like DeepFace, but modifies the loss function to encourage more discriminative face embeddings.

### 3.2 Face Identity, Verification, and Recognition

- **Face identity (training target):** Learns what the facial features are for a person based on/given a dataset;
- **Face verification (1:1 matching):** Determines if two face images belong to the same person;
- **Face recognition (1:N identification, evaluation task):** Determines who a particular person is among a database of *known* identities using a new picture.

### 3.3 Evaluation Metrics

- **Identification rate:** The percentage of faces the model correctly identifies from a database of known people.
- **TAR (True Acceptance Rate):** The percentage of same-person pairs correctly accepted as the same person. **TAR + FRR** (False Rejection Rate) **= 100%**.
- **FAR (False Acceptance Rate):** The percentage of different-person pairs incorrectly accepted as the same person. **FAR + TRR** (True Rejection Rate) **= 100%**.

### 3.4 Margin-Based Methods

**Margin-based methods:** training approaches that modify the *loss function* to require a larger separation between different identities in embedding space.


## 4. Vision Transformer / ViT — Dosovitskiy et al. (2021)

**Paper:** [An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

### 4.1 Key Questions to Understand

Paid specific attention to: **How an image becomes patches/tokens**; patch embeddings + positional embeddings + self-attention; **CNN vs Transformer**; why ViT initially needed large-scale pretraining

- Unlike the previous three papers, ViT focuses on general image understanding, not specifically facial recognition.

### 4.2 Datasets Used

- **ImageNet-21k:** ~14 million images across 21k object categories; used for large-scale supervised pre training;
- **JFT-300M:** ~300 million images with noisy labels across thousands of categories; even larger scale supervised pre training;
- **ImageNet-1k:** ~1.3 million training images across 1k categories; Image classification evaluation and fine-tuning;
- **CIFAR-10/100:** 60k small mages across 10 or 100 categories; testing performance on smaller classification datasets;
- **VTAB:** A benchmark suite covering 19 diverse visual tasks; testing whether pre training representations (features learned from one dataset) transfer to different tasks **[transfer learning]**;

### 4.3 Training Types

- **Training:** A general term for teaching a model to perform a task.
- **Pretraining:** The initial training stage, where a model learns broadly useful features before being adapted to a more specific task.
- **Fine-tuning:** Additional training that adapts the pretrained model to that specific task.

### 4.4 Transformer and Attention Mechanism

**Transformer:** A type of neural network that uses **attention mechanisms** to learn relationships between different parts of its input. Unlike CNNs, which primarily examine local regions, transformers can directly relate distant parts of an image.

- **Attention mechanism:** a technique that allows a NN to determine which parts of an input image (not just locally) are most relevant to one another and give them different weights when processing information.
- E.g., When analyzing a face, a ViT can learn relationships between the eyes, nose, and mouth, even though they occupy different image regions.

### 4.5 Self-Attention and Multi-Head Self-Attention

- **Self-attention:** A mechanism that lets each part of an image determine how much information to gather from other parts.
- **Multi-head self-attention:** Instead of calculating attention just once, the model uses multiple attention heads in parallel, allowing it to learn different relationships simultaneously;
  - E.g., one head might focus on relationships between facial features, while another focuses on broader visual patterns. These specializations are learned, not predefined.

### 4.6 Significance of This Paper

Significance of this paper: it changes what **"image representation"** means, using ViT instead of CNN.

- **CNN:** local filters gradually build features;
- **ViT:** divide the image into patches and learn relationships between them using attention.


## 5. DINO — Caron et al. (2021)

**Paper:** *Emerging Properties in Self-Supervised Vision Transformers*

> **Paid specific attention to:** Self-supervised learning; teacher–student/self-distillation idea; why multiple views of the same image are used; what representation is learned without labels; why meaningful spatial/attention structure emerges.

### 5.1 Semantic Structure

- **Semantic structure:** Meaningful relationships within an image, such as which pixels belong to the same object or which regions represent different objects.
  - E.g., DINO is trained on images without being told where the objects are, yet its learned representations can reveal the boundaries between a dog and its background.
  - **Why evaluate this?** Researchers want to see whether DINO has learned meaningful visual concepts, rather than just patterns useful for image classification. This is especially interesting because it learns without human-provided labels.
  - **Key takeaway:** Meaningful visual structure can emerge from self-supervised learning, making its representations potentially useful for facial and perceptual analysis.

### 5.2 Self-Supervised Learning

- Learning patterns useful for image classification does not necessarily require human-provided labels.
  - **Supervised learning** (e.g., original ViT): Uses labelled images (e.g., "dog," "cat") to learn features that help classify objects.
  - **Self-supervised learning** (e.g., DINO): Learns visual features without human labels, but it is still given a specific learning objective. For DINO, this involves learning representations from different modified views of the same image.
  - In other words, self-supervised learning is NOT simply letting the model learn whatever it wants. Researchers still design the learning task; they just don't need humans to label the training images.

- Self-supervised learning is a form of **unsupervised learning**, but the reverse is not always true. For example, K-means clustering:
  - **K-means:** Groups similar data points into clusters. *Grouping is the task.*
  - **DINO:** Learns useful visual representations by training different views of the same image to produce consistent outputs. *Learning these representations is the task.*

- **Key distinction:** Labels are not required to learn useful visual representations, although labels are often used later to train or evaluate a classifier.

### 5.3 Input Representation

- DINO uses different modified (e.g., augmented/cropped) versions of the same image to teach the model to recognize consistent visual features despite changes in appearance.

### 5.4 Method — Self-Distillation

- DINO uses two neural networks during training: **teacher and student networks**.
  - **Teacher network:** Produces a probability distribution from one view of an image.
  - **Student network:** Processes another view of the same image and learns to match the teacher's probability distribution. This is the *learning objective* of DINO.

- **Why use a probability distribution?** It expresses how strongly the network associates an image with different learned visual patterns.

- **Distillation:** Transferring knowledge from a teacher model to a student model.
- **Self-distillation:** The student learns by adjusting its *weights* to match the teacher's predictions, and the teacher is updated using a **moving average** of the student's weights.
  - **Key point:** The teacher and student *develop together* during training. The direction of influence is both ways, but only the student is directly trained by minimizing the loss function.
  - The teacher isn't necessarily more "knowledgeable" than the student, but it provides more **stable** probability distributions because its weights are smoothed using the moving average.

### 5.5 Evaluation

#### K-Nearest Neighbours (k-NN)

- Converts an image into an embedding, then finds the *k* most similar embeddings from labelled examples.
- A prediction is made about the image's category based on its nearest neighbours.

#### Linear Probing

- Freezes DINO's pretrained network (i.e., doesn't change its weights).
- Trains a simple linear classifier using the embeddings and labelled examples.
- The classifier is tested on whether it can correctly distinguish categories in new images.

**Key point:** Both methods evaluate how useful DINO's learned representations are.

#### Qualitative and Quantitative Evaluation

Researchers evaluate whether DINO's attention patterns reveal meaningful objects and regions within images using two approaches:

- **Qualitative (visual inspection):** Researchers look at attention maps to see whether DINO naturally focuses on meaningful regions, such as a dog's body rather than the background.
- **Quantitative (numerical measurement):** Researchers measure how accurately those regions match actual object boundaries using metrics such as **IoU (Intersection over Union)**, which measures how much two regions overlap.
- **Example:** If DINO highlights a dog in an image, researchers can visually inspect the highlighted area (qualitative) and calculate how closely it matches the dog's actual outline (quantitative).
- **Why?** This tells researchers whether DINO has learned meaningful visual structure *without* being explicitly trained to identify object boundaries.

### 5.6 Attention Maps

- An **attention map** is a visual representation showing how strongly different parts of an image receive attention from the model. It's a heatmap generated from attention weights.
  - E.g., in a picture of a dog standing on grass, an attention map might highlight the dog's body more strongly than the surrounding grass.
  - In ViT, researchers can visualize how strongly the classification token attends to different image patches.

### 5.7 Limitations

- The teacher–student training design requires additional memory and computation because there are two networks.
- Researchers must carefully configure the teacher's moving-average updates, prediction distributions, and techniques that prevent representation collapse.

---

## 6. AdaFace — Kim, Jain & Liu (2022)

**Paper:** *AdaFace: Quality Adaptive Margin for Face Recognition*

> **Paid specific attention to:** Image quality as a learning problem; why low-quality ≠ useful hard example; how quality-adaptive margins modify face learning.

### 6.1 Image Quality and Feature Norm

- AdaFace uses the **norm (magnitude) of the network's feature vector** as an indirect estimate of face-image quality.
  - A larger feature norm generally indicates higher recognizability.
  - A smaller feature norm generally indicates lower recognizability.
  - This is a correlation, not a guaranteed measure of image quality.

### 6.2 Quality-Adaptive Margin

- **Quality-adaptive margin:** AdaFace adjusts how it trains based on estimated image quality.
  - **Margin loss:** A *loss function* that calculates a training penalty when the model's predictions do not satisfy a desired separation between identities.
  - The penalty is used through backpropagation to adjust the neural network's weights, helping it distinguish different people's faces more clearly.
