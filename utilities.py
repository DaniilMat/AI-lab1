import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y):
    # Определение границ графика
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Создание сетки координат
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    # Предсказание классов для каждой точки сетки
    Z = classifier.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )
    Z = Z.reshape(xx.shape)

    # Отображение границ классов
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Отображение исходных точек
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap='viridis',
        edgecolors='black',
        s=80
    )

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(alpha=0.2)
