import numpy as np
import matplotlib.pyplot as plt

def decision_boundary(X, y, model, title="Decision Boundary", xlabel="Feature 1", ylabel="Feature 2"):
    #minimum and maximum values for the x and y 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #mesh grid 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    #predict the class labels for each point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    #contour areas based on predicted class labels (decision boundary)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    for class_label in np.unique(y):
        plt.scatter(X[y == class_label][:, 0], X[y == class_label][:, 1], edgecolors='k', label=f'Class {class_label}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

#decision_boundary(X, y, clf, title="Decision Boundary for Iris Dataset (Two Features)", xlabel="Feature 1", ylabel="Feature 2")
