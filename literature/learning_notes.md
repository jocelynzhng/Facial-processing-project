
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
