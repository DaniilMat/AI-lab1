import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Загрузка Boston Housing из OpenML
data = fetch_openml(
    name='boston',
    version=1,
    as_frame=False,
    parser='liac-arff'
)

X = np.asarray(data.data, dtype=float)
y = np.asarray(data.target, dtype=float)

# Перемешивание данных
X, y = shuffle(X, y, random_state=7)

# Разделение на обучающую и тестовую выборки
num_training = int(0.8 * len(X))

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# тестовые точки
test_points = [
    [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27],
    [0.1, 25, 5, 0, 0.4, 6.5, 35, 6, 5, 300, 15, 390, 5],
    [12, 0, 18, 0, 0.6, 6.2, 60, 3, 20, 500, 18, 380, 10],
    [0.5, 20, 6, 0, 0.45, 7.2, 30, 5, 4, 250, 16, 395, 4]
]

# Параметры экспериментов
experiments = [
    (1.0, 0.1),
    (10.0, 0.1),
    (100.0, 0.5)
]

# Обучение и оценка моделей
for C, epsilon in experiments:
    regressor = make_pipeline(
        StandardScaler(),
        SVR(kernel='linear', C=C, epsilon=epsilon)
    )

    regressor.fit(X_train, y_train)

    y_test_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)

    print(f"\n#### C = {C}, epsilon = {epsilon} ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    for i, point in enumerate(test_points, start=1):
        prediction = regressor.predict([point])[0]
        print(f"Точка {i}: прогнозируемая стоимость = {prediction:.2f}")
