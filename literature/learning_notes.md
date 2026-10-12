
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
