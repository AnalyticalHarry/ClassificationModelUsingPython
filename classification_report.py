import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def classification_report(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")
    if not all(np.isin(y_true, [0, 1])) or not all(np.isin(y_pred, [0, 1])):
        raise ValueError("y_true and y_pred must be binary (0 and 1).")
    
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    print("True Positives:", true_positive)
    print("True Negatives:", true_negative)
    print("False Positives:", false_positive)
    print("False Negatives:", false_negative)
    print()
    confusion_mat = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    accuracy = (true_positive + true_negative) / len(y_true)
    precision_0 = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    recall_0 = true_negative / (true_negative + false_negative) if (true_negative + false_negative) != 0 else 0
    specificity_0 = recall_0
    precision_1 = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall_1 = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    specificity_1 = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    
    print("Confusion Matrix:")
    print(confusion_mat)
    print("\nClassification Metrics:")
    print("Accuracy:", accuracy)
    print("Precision Class 0:", precision_0)
    print("Recall (Sensitivity) Class 0:", recall_0)
    print("Specificity Class 0:", specificity_0)
    print("Precision Class 1:", precision_1)
    print("Recall (Sensitivity) Class 1:", recall_1)
    print("Specificity Class 1:", specificity_1)
    print("False Positives:", false_positive, "cases")
    print("False Negatives:", false_negative, "cases")
    print("Correctly Predicted:", true_positive + true_negative, "cases")
    print("Wrong Predicted:", false_positive + false_negative, "cases")

    metrics_labels = ["Precision", "Recall (Sensitivity)", "Specificity"]
    metrics_values_0 = [precision_0, recall_0, specificity_0]
    metrics_values_1 = [precision_1, recall_1, specificity_1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Create two subplots
    bar_width = 0.35
    index = np.arange(len(metrics_labels))
    bar1 = ax1.bar(index, metrics_values_0, bar_width, label='Class 0', alpha=0.7, color='red')
    bar2 = ax1.bar(index + bar_width, metrics_values_1, bar_width, label='Class 1', alpha=0.7, color='blue')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Scores')
    ax1.set_title('Classification Metrics')
    ax1.grid(True, ls='--', color='black', alpha=0.5)
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(metrics_labels)

    # bars with their values
    for bar, value in zip(bar1, metrics_values_0):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.2f}', ha='center', va='bottom', color='red')
    for bar, value in zip(bar2, metrics_values_1):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.2f}', ha='center', va='bottom', color='blue')

    # heatmap for the confusion matrix
    heatmap = ax2.imshow(confusion_mat, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(heatmap, ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax2.set_yticklabels(['Actual 0', 'Actual 1'])

    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, confusion_mat[i, j],
                            ha='center', va='center', color='white', fontsize=12)

    ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
    plt.show()