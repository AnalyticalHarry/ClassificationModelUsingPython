import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def decision_boundaries(X, y, model, elevations=[10, 30, 60, 70, 80, 30], azimuths=[20, 45, 70, 95, 120, 30], class_labels=None):
    def decision_boundary_3d(X, y, model, ax, elev=20, azim=30, title="Decision Boundary", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3", class_labels=None):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        #elevation and azimuth
        ax.view_init(elev=elev, azim=azim)
        if class_labels is None:
            class_labels = [f"Class {i}" for i in range(len(np.unique(y)))]
        #classes
        unique_classes = np.unique(y)
        for i, class_label in enumerate(class_labels):
            ax.scatter(X[y == unique_classes[i]][:, 0], X[y == unique_classes[i]][:, 1], X[y == unique_classes[i]][:, 2], label=class_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.set_title(title)

    fig, axes = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(18, 12)) 
    axes = axes.flatten()
    for ax, elev, azim in zip(axes, elevations, azimuths):
        decision_boundary_3d(X, y, model, ax, elev=elev, azim=azim, title=f"SVM Decision Boundary (Elev={elev}, Azim={azim})", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3", class_labels=class_labels)
    plt.show()


#class_labels = ["Setosa", "Versicolor", "Virginica"]
#decision_boundaries(X, y, clf, class_labels=class_labels)