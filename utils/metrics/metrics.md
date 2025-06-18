Structured summary of **tracking and association metrics** including their **titles, mathematical formulas**, and **brief descriptions**. This version of the metrics assumes that **ground truth labels <u>***do not***</u> contain object IDs**, which results in the omission of traditional ID-based metrics like ID switches, true track assignment, and fragmentation accuracy per identity.

---

### 🔍 **Detection-Level Metrics (Association Between Predictions and Ground Truth)**

| **Title**     | **Formula**                                       | **Description**                                                   |
| ------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| **Precision** | `TP / (TP + FP)`                                  | Measures the proportion of predicted detections that are correct. |
| **Recall**    | `TP / (TP + FN)`                                  | Measures the proportion of actual objects correctly detected.     |
| **F1 Score**  | `2 * (Precision * Recall) / (Precision + Recall)` | Harmonic mean of precision and recall, balancing both.            |
| **DetA**                    | `(TP - FP) / GT`              | Detection Tracking Accuracy - detection quality with false positive penalty |

> **Note:** These are computed by associating predictions and ground truth using Euclidean distance and the Hungarian algorithm. Since no object IDs are used, the association is spatial only.

> **Note:** DetA (Should be implemented)
#### Why DetA is Perfect for This Use Case

1. **No Track IDs Required**: DetA works with frame-by-frame detection associations
2. **Penalizes False Positives**: Unlike recall, DetA accounts for over-detection
3. **Tracking-Oriented**: Designed specifically for tracking evaluation scenarios
4. **Bounded Range**: DetA ranges from -∞ to 1.0, where 1.0 is perfect performance
5. **Interpretable**: Negative values indicate more false positives than true positives

DetA essentially answers: "How well are we detecting objects while avoiding false alarms?" which is exactly what you need for radar tracking evaluation without track identity information.


---

### 📏 **Distance-Based Metrics (For Assigned Matches Only)**

| **Title**                   | **Formula**                   | **Description**                                                           |
| --------------------------- | ----------------------------- | ------------------------------------------------------------------------- |
| **Mean Euclidean Distance** | `mean(distances ≤ threshold)` | Average Euclidean distance between matched pairs under a given threshold. |
| **Std. Euclidean Distance** | `std(distances ≤ threshold)`  | Spread of distances for matched pairs.                                    |
| **Min Distance**            | `min(distances ≤ threshold)`  | Minimum of matched pair distances.                                        |
| **Max Distance**            | `max(distances ≤ threshold)`  | Maximum of matched pair distances.                                        |
| **Mean IoU (Intersection over Union)** | mean(IoU for matched pairs) | Average IoU of matched bounding boxes under the association threshold. |

> **Note:** Only includes valid associations under a distance threshold. IoU is only applicable when bounding boxes are available.

---

### 📊 **Aggregated Multi-Frame Metrics**

| **Title**                                    | **Formula**                                        | **Description**                                                  |
| -------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| **MOTA (Multiple Object Tracking Accuracy)** | `1 - (FP + FN) / GT`                               | Penalizes missed detections and false positives over all frames. |
| **MOTP (Precision)**                         | `mean(Euclidean distance of correct associations)` | Measures localization precision of the tracker.                  |

> **Note:** These are computed from per-frame results. Since there are **no object IDs**, **ID switches** are not tracked, affecting true MOTA interpretability.


---

### 🚫 **Fragmentation and ID-Based Metrics (Skipped Due to Missing IDs)**

| **Title**                       | **Reason Not Included**                                    |
| ------------------------------- | ---------------------------------------------------------- |
| **ID Switches**                 | Cannot be computed without object IDs.                     |
| **Fragmentation Rate**          | Requires continuous identity tracking.                     |
| **Track Purity / Completeness** | Require comparing actual ID sequences with predicted ones. |
| **Temporal Consistency Score**  | Not implemented without IDs.                               |

---

### 🧮 **General Statistics**

| **Title**                             | **Description**                           |
| ------------------------------------- | ----------------------------------------- |
| **Total Associations / TP / FP / FN** | Number of associations across all frames. |

---

### 📌 Important Notes on Changes

* **No Ground Truth IDs**: Since object identities are not available in the labels, all ID-based metrics like **ID switches**, **track purity**, **fragmentation**, and **lifecycle analysis** are either skipped or placeholders.
* **Associations are purely spatial**: Matching is performed via distance thresholds (Hungarian algorithm on Euclidean distance), not identity continuity.
* **Distance threshold is a key hyperparameter**: It defines what is considered a valid match for all detection/tracking evaluations.

