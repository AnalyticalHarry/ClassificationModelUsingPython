import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Calculate and display a confusion matrix along with classification metrics.

    Args:
    - y_true (numpy.ndarray): True labels (ground truth).
    - y_pred (numpy.ndarray): Predicted labels.

    Returns:
    None

    Usage:
    >>> confusion_matrix(y_true, y_pred)

    This function calculates and displays the following:
    - Confusion Matrix
    - Accuracy: A measure of overall model correctness.
    - Precision: Proportion of true positive predictions out of all positive predictions.
    - Recall (Sensitivity): Proportion of true positive predictions out of all actual positives.
    - Specificity: Proportion of true negative predictions out of all actual negatives.
    - Type 1 Error (False Positives): Incorrectly predicted positive cases.
    - Type 2 Error (False Negatives): Incorrectly predicted negative cases.
    - Correctly Predicted: Total correct predictions.
    - Wrong Predicted: Total incorrect predictions.

    Note:
    - In binary classification, 'positive' typically refers to the class labeled as 1, and 'negative' to the class labeled as 0.
    - A higher accuracy indicates a better model, but consider other metrics for imbalanced datasets.
    - Precision, Recall, and Specificity provide insights into model performance.

    """
    #length of y_true and y_pred is the same
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")
    
    #y_true and y_pred are binary (0 and 1)
    if not all(np.isin(y_true, [0, 1])) or not all(np.isin(y_pred, [0, 1])):
        raise ValueError("y_true and y_pred must be binary (0 and 1).")

    #true positive cases (actual 1, predicted 1)
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    #true negative cases (actual 0, predicted 0)
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    #false positive cases (actual 0, predicted 1)
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    #false negative cases (actual 1, predicted 0)
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    #Type 1 error (false positives)
    type_1_error = false_positive
    #Type 2 error (false negatives)
    type_2_error = false_negative
    #correct predictions
    correct_predictions = true_positive + true_negative
    #wrong predictions
    wrong_predictions = false_positive + false_negative

    #confusion matrix
    confusion_mat = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    #division by zero
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    specificity = true_negative / (true_negative + false_negative) if (true_negative + false_negative) != 0 else 0

    #confusion matrix and classification metrics
    print("Confusion Matrix:")
    print(confusion_mat)
    print("\nClassification Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall (Sensitivity):", recall)
    print("Specificity:", specificity)
    
    #errors 
    print("\nAdditional Information:")
    print("Type 1 Error (False Positives):", type_1_error, "cases")
    print("Type 2 Error (False Negatives):", type_2_error, "cases")
    print("Correctly Predicted:", correct_predictions, "cases")
    print("Wrong Predicted:", wrong_predictions, "cases")
    
    
    #automated result 
    if accuracy >= 0.80:
        print("Guidance: High accuracy (80% to 100%) suggests excellent model performance.")
    elif accuracy >= 0.75:
        print("Guidance: Good accuracy (75% to 80%) indicates a solid model performance.")
    elif accuracy >= 0.60:
        print("Guidance: Consider reviewing the model (60% to 75%) as the accuracy is below the recommended threshold.")
    else:
        print("Guidance: The accuracy is below 60%. Strongly consider reviewing and improving the model.")
    
    if precision >= 0.80:
        print("Guidance: High precision indicates a low false positive rate.")
    else:
        print("Guidance: Precision is below the recommended threshold (>= 0.80). Check for false positives.")
    
    if recall >= 0.80:
        print("Guidance: High recall indicates a low false negative rate.")
    else:
        print("Guidance: Recall is below the recommended threshold (>= 0.80). Check for false negatives.")
    
    if specificity >= 0.80:
        print("Guidance: High specificity suggests a good ability to identify true negatives.")
    else:
        print("Guidance: Specificity is below the recommended threshold (>= 0.80). Check for false negatives.")

