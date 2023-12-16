#confusion matrix 
def confusion_matrix(y_true, y_pred):
    #true positive cases (actual 1, predicted 1)
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    #true negative cases (actual 0, predicted 0)
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    #false positive cases (actual 0, predicted 1)
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    #false negative cases (actual 1, predicted 0)
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    #2x2 confusion matrix
    confusion_mat = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    #accuracy (correct predictions divided by total predictions)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    #precision (true positives divided by true positives plus false positives)
    precision = true_positive / (true_positive + false_positive)
    #recall (sensitivity, true positives divided by true positives plus false negatives)
    recall = true_positive / (true_positive + false_negative)
    #specificity (true negatives divided by true negatives plus false negatives)
    specificity = true_negative / (true_negative + false_negative)
    
    #confusion matrix and classification metrics
    print("Confusion Matrix:")
    print(confusion_mat)
    print("\nMetrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall (Sensitivity):", recall)
    print("Specificity:", specificity)
