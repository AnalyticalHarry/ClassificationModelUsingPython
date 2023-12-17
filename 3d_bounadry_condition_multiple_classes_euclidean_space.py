import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

#decision boundary with convex hulls
def decision_boundary_3d(X, y, model, title="Decision Boundary", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3", class_labels=None, elev=90, azim=30):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', elev=elev, azim=azim)
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(len(np.unique(y)))]
    #convex hull for each class
    unique_classes = np.unique(y)
    for i, class_label in enumerate(class_labels):
        class_points = X[y == unique_classes[i]]
        ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], label=class_label)
        if len(class_points) > 3:  
            hull = ConvexHull(class_points)
            for simplex in hull.simplices:
                ax.plot(class_points[simplex, 0], class_points[simplex, 1], class_points[simplex, 2], color="k", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

class_labels = ["Setosa", "Versicolor", "Virginica"]

decision_boundary_3d(X, y, clf, title="Decision Boundary", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3", class_labels=class_labels, elev=30, azim=65)
