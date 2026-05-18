# Member 2 — K-Means Clustering & Evaluation

## 1. Implementation Details
The K-Means clustering algorithm was implemented from scratch following Lloyd's algorithm. The key steps of our implementation include:
*   **Initialization**: Centroids are randomly sampled from the input training data instances to ensure they start within the data distribution.
*   **Distance Metric**: We broadcast Euclidean distance computations across the data matrix and the $K$ centroids using `numpy` efficiently.
*   **Empty Cluster Handling**: If a centroid loses all mapped points during an iteration, it is randomly reassigned to another data point to maintain $K$ active clusters.
*   **Convergence**: The algorithm halts either when it reaches `MAX_ITER_KM` (set to 300) or when the shift in centroid placement falls below a threshold `tolerance` (set to 1e-4).

## 2. Design Decisions & Assumptions

### Cluster to Label Mapping Strategy
Because K-Means is an unsupervised algorithm, it groups data into arbitrary cluster IDs (e.g., $0$ to $K-1$). We needed a mapping approach to evaluate standard classification metrics against the 40 known subjects. 

We used a **Dual-Strategy approach** dependent on the value of $K$:
1.  **Hungarian Algorithm ($K=40$)**: When $K$ perfectly equals the true number of subjects ($N=40$), we want to extract to strict 1-to-1 matching. We calculate an overlap cost matrix and apply `scipy.optimize.linear_sum_assignment` which optimally assigns exactly one cluster to one true identity.
2.  **Majority Vote ($K \in \{20, 60\}$)**: A strict 1-to-1 match is impossible here. If $K=20$ (under-clustering), finding a 1-to-1 match would leave 20 subjects completely unidentified. If $K=60$ (over-clustering), the algorithm would fail. Instead, each discovered cluster takes the identity of the most frequent true subject contained within it.

## 3. Experimental Results
We ran the K-Means algorithm over combinations of PCA thresholds $\alpha \in \{0.8, 0.85, 0.9, 0.95\}$ and cluster counts $K \in \{20, 40, 60\}$. 

**Training Accuracy Tabulation:**

| Variance Retained ($\alpha$) | $K=20$ | $K=40$ | $K=60$     |
| ---------------------------- | ------ | ------ | ---------- |
| $\alpha=0.80$ (36 comps)     | 0.4150 | 0.6700 | 0.8050     |
| $\alpha=0.85$ (51 comps)     | 0.4100 | 0.6600 | **0.8200** |
| $\alpha=0.90$ (76 comps)     | 0.4100 | 0.6450 | 0.7900     |
| $\alpha=0.95$ (115 comps)    | 0.4000 | 0.6450 | 0.7800     |

*(Note: Plotted figures `kmeans_acc_vs_k.png` and `kmeans_acc_vs_alpha.png` were successfully generated and deposited in the `outputs/` directory for visual reference).*

## 4. Analysis Questions

**Can you find a relation between $\alpha$ and classification accuracy?**
Yes. From the results, accuracy generally peaks around $\alpha=0.80$ to $\alpha=0.85$ and then begins to drop off at $\alpha=0.90$ and $\alpha=0.95$. While higher $\alpha$ values theoretically retain more variance from the original dataset, the lower-energy principal components often represent idiosyncratic noise or lighting variations rather than core facial features. Because K-Means relies on pure Euclidean distance, introducing dozens of extra noisy dimensions (the "curse of dimensionality") actively skews the centroid distances and degrades clustering performance.

**Can you find a relation between $K$ and classification accuracy?**
There is a direct correlation: as $K$ increases, mapping accuracy increases. 
*   **$K=20$**: Forcefully merges 40 distinct subjects into 20 clusters, heavily penalizing accuracy (caps around $\sim 41\%$) because half the subjects functionally disappear.
*   **$K=40$**: Reaches a fair assessment ($\sim 65\% - 67\%$) reflecting the realistic clustering potential on 40 distinct identities.
*   **$K=60$**: Mechanically artificially inflates accuracy (exceeding $80\%$). By allowing over-segmentation, subjects are broken down into purer "micro-clusters", and our Majority Vote maps those pure clusters directly back to the true subjects. This naturally reduces error but comes at the cost of over-partitioning the data representation.

## 5. Evaluation of the Best Model
The optimal combination found during training was **$\alpha=0.85$ and $K=60$**.
Applying this trained model and its mapping dictionary to the unseen **Test Set** yielded the following outstanding final performance:

*   **Test Accuracy**: 0.8200 (82.0%)
*   **Test Macro F1-Score**: 0.8026

The Test Accuracy perfectly aligned with the Training Accuracy, displaying a well-generalized mapping that successfully recognized unseen variations within the test images. A comprehensive **Confusion Matrix** (`confusion_matrix.png`) reflecting the exact distribution of False Positives/Negatives across the 40 subjects has been exported to the `outputs/` folder.