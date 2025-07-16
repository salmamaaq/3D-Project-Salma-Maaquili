
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_classification_metrics(predictions, true_labels):
    """
    Calculates classification metrics (accuracy, precision, recall, f1-score).
    Args:
        predictions (torch.Tensor): Predicted class labels (e.g., from argmax of model output).
        true_labels (torch.Tensor): True class labels.
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and f1-score.
    """
    predictions_np = predictions.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    accuracy = accuracy_score(true_labels_np, predictions_np)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels_np, predictions_np, average='weighted', zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    return metrics

if __name__ == '__main__':
    # Example usage
    # Simulate model output (logits) and true labels
    batch_size = 2
    num_classes = 40

    # Example 1: Perfect prediction for one sample
    logits1 = torch.randn(batch_size, num_classes) # Raw model output
    predictions1 = torch.argmax(logits1, dim=1) # Predicted class IDs
    true_labels1 = torch.randint(0, num_classes, (batch_size,)) # True class IDs

    print(f"\nExample 1: Predictions: {predictions1}, True Labels: {true_labels1}")
    metrics1 = calculate_classification_metrics(predictions1, true_labels1)
    print(f"Metrics: {metrics1}")

    # Example 2: More complex scenario
    logits2 = torch.tensor([
        [0.1, 0.9, 0.0, 0.0], # Predicts class 1
        [0.8, 0.1, 0.1, 0.0], # Predicts class 0
        [0.0, 0.0, 0.9, 0.1]  # Predicts class 2
    ])
    predictions2 = torch.argmax(logits2, dim=1)
    true_labels2 = torch.tensor([1, 0, 3]) # True labels

    print(f"\nExample 2: Predictions: {predictions2}, True Labels: {true_labels2}")
    metrics2 = calculate_classification_metrics(predictions2, true_labels2)
    print(f"Metrics: {metrics2}")



