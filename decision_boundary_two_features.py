import numpy as np
import matplotlib.pyplot as plt

def decision_boundary(X, y, model, title="Decision Boundary", xlabel="Feature 1", ylabel="Feature 2", class_labels=("Class 0", "Class 1")):
    #minimum and maximum values for the x and y axes
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #mesh grid points for the decision boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    #predict class labels for each point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(7, 5))
    #contour
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #class 0 and class 1
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', cmap=plt.cm.coolwarm, edgecolors='k', label=class_labels[0])
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', cmap=plt.cm.coolwarm, edgecolors='k', label=class_labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

# decision_boundary(X, y, clf, title="SVM Decision Boundary for Iris Dataset", xlabel="Sepal Length", ylabel="Sepal Width")
