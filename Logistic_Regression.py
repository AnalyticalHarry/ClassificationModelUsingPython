import numpy as np
from IPython.display import display, HTML

def tab(data):
    if isinstance(data, list) and all(isinstance(d, dict) for d in data):
        columns = list(data[0].keys())
        rows = [list(row.values()) for row in data]
    elif isinstance(data, list) and all(isinstance(d, list) for d in data):
        columns = data[0]
        rows = data[1:]
    else:
        raise ValueError("Input data should be a list of dictionaries or a list of lists.")

    table_html = "<table>"
    table_html += "<tr>"
    for col in columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr>"

    for row in rows:
        table_html += "<tr>"
        for col in row:
            table_html += f"<td>{col}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    display(HTML(table_html))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_confusion_matrix(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[true_negative, false_positive], [false_negative, true_positive]])

def print_confusion_matrix(set_name, y_true, y_pred):
    conf_matrix = calculate_confusion_matrix(y_true, y_pred)
    conf_matrix_data = [
        {'': '', 'Predicted 0': 'True 0', 'Predicted 1': 'True 1'},
        {'': 'True 0', 'Predicted 0': conf_matrix[0, 0], 'Predicted 1': conf_matrix[0, 1]},
        {'': 'True 1', 'Predicted 0': conf_matrix[1, 0], 'Predicted 1': conf_matrix[1, 1]}
    ]

    print(f"Confusion Matrix ({set_name}):")
    tab(conf_matrix_data)

def print_summary_statistics(conf_matrix, set_name):
    TP, TN, FP, FN = conf_matrix[0, 0], conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0]
    correctly_predicted = TP + TN
    falsely_predicted = FP + FN
    
    print(f"{set_name} Set:")
    print(f'True Positives: {TP}')
    print(f'True Negatives: {TN}')
    print(f'False Positives: {FP}')
    print(f'False Negatives: {FN}')
    print(f'Total Correctly Predicted: {correctly_predicted}')
    print(f'Total Falsely Predicted: {falsely_predicted}\n')
    
def print_classification_metrics(set_name, y_true, y_pred):
    conf_matrix = calculate_confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = conf_matrix[0, 0], conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0]
    
    print()
    #classification Accuracy
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    print(f'Classification Accuracy ({set_name}): {round(classification_accuracy, 4)}')

    #classification Error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print(f'Classification Error ({set_name}): {round(classification_error, 4)}')

    #precision
    precision = TP / float(TP + FP) if TP + FP != 0 else 0.0
    print(f'Precision ({set_name}): {round(precision, 4)}')

    #recall
    recall = TP / float(TP + FN) if TP + FN != 0 else 0.0
    print(f'Recall ({set_name}): {round(recall, 4)}')

    #True Positive Rate
    true_positive_rate = TP / float(TP + FN) if TP + FN != 0 else 0.0
    print(f'True Positive Rate ({set_name}): {round(true_positive_rate, 4)}')

    #False Positive Rate
    false_positive_rate = FP / float(FP + TN) if FP + TN != 0 else 0.0
    print(f'False Positive Rate ({set_name}): {round(false_positive_rate, 4)}')

    #True Negative Rate (specificity)
    specificity = TN / float(TN + FP) if TN + FP != 0 else 0.0
    print(f'Specificity ({set_name}): {round(specificity, 4)}')

    
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit_logistic_regression(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return np.round(predictions)