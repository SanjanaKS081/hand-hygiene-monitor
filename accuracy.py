import json

def calculate_metrics(data):
    """
    Calculates accuracy, precision, recall, and F1 score from detection data.
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    tn = 0  # True Negatives
    fn = 0  # False Negatives

    for frame in data:
        has_detection = len(frame['detections']) > 0
        has_hand = len(frame['hands']) > 0

        if has_hand and has_detection:
            tp += 1
        elif not has_hand and has_detection:
            fp += 1
        elif has_hand and not has_detection:
            fn += 1
        elif not has_hand and not has_detection:
            tn += 1

    # Calculate metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def main():
    file_path = 'detections.json'
    try:
        with open(file_path, 'r') as file:
            detections_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    accuracy, precision, recall, f1_score = calculate_metrics(detections_data)

    print("REAL-TIME SYSTEM PERFORMNACE:")
    print("-" * 30)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1_score * 100:.2f}%")

if __name__ == "__main__":
    main()