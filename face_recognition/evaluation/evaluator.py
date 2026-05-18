import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, confusion_matrix

class Evaluator:
    @staticmethod
    def map_clusters_to_labels_majority_vote(cluster_labels, true_labels):
        """
        Maps each cluster to the most frequent true label in it.
        """
        mapped_labels = np.zeros_like(cluster_labels)
        for cluster_id in np.unique(cluster_labels):
            mask = (cluster_labels == cluster_id)
            if np.sum(mask) > 0:
                majority_label = np.bincount(true_labels[mask]).argmax()
                mapped_labels[mask] = majority_label
        return mapped_labels

    @staticmethod
    def map_clusters_to_labels_hungarian(cluster_labels, true_labels): # Might get deleted
        max_cluster_id = int(np.max(cluster_labels)) + 1
        max_true_id = int(np.max(true_labels)) + 1
        
        overlap_matrix = np.zeros((max_cluster_id, max_true_id), dtype=np.int64)
        for c, t in zip(cluster_labels, true_labels):
            overlap_matrix[c, t] += 1
            
        # we minimize (max_overlap - overlap_matrix) to avoid negative costs.
        row_ind, col_ind = linear_sum_assignment(overlap_matrix.max() - overlap_matrix)
        
        # Create mapping dictionary mapping cluster_id -> true_label_id
        mapping = {r: c for r, c in zip(row_ind, col_ind)}
        
        mapped_labels = np.array([mapping.get(c, -1) for c in cluster_labels])
        return mapped_labels

    @staticmethod
    def calculate_accuracy(mapped_predictions, true_labels):
        return np.sum(mapped_predictions == true_labels) / len(true_labels)
        
    @staticmethod
    def evaluate_test_set(mapped_predictions, true_labels):
        accuracy = Evaluator.calculate_accuracy(mapped_predictions, true_labels)
        f1 = f1_score(true_labels, mapped_predictions, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, mapped_predictions)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm
        }
