def decision_boundary(X, y, model, title="Decision Boundary", xlabel="Feature 1", ylabel="Feature 2", 
                      resolution=0.01, cmap_decision=plt.cm.coolwarm, 
                      grid=False, figsize=(7, 5)):
    #minimum and maximum values for the features
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #meshgrid for the feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

    #predict class for each point in the meshgrid using the model 
    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    except AttributeError:
        raise ValueError("Model provided does not support 'predict' method")
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=figsize)
    #decision boundary using contour plot
    plt.contourf(xx, yy, Z, cmap=cmap_decision, alpha=0.8)
    unique_classes = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    for i, class_label in enumerate(unique_classes):
        plt.scatter(X[y == class_label][:, 0], X[y == class_label][:, 1], 
                    color=colors[i], edgecolors='k', label=f'Class {class_label}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + ('' if not hasattr(model, '__name__') else f' - {model.__name__}'))
    plt.legend(loc='best')
    if grid:
        plt.grid(True)
    plt.show()

#decision_boundary(X, y, clf, title="model name", xlabel="Feature 1", ylabel="Feature 2")