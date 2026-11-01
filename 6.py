import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

# Входной файл
input_file = 'Lab01_data\income_data.txt'

# Чтение данных
X = []
y = []

count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f:
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line.strip().split(', ')

        # Удаление точки после метки, если она присутствует
        data[-1] = data[-1].rstrip('.')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Преобразование в numpy array
X = np.array(X)

# Кодирование строковых данных
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y,
    test_size=0.2,
    random_state=5,
    stratify=y
)

# Создание и обучение классификатора
classifier = OneVsOneClassifier(
    LinearSVC(random_state=0, max_iter=10000)
)

classifier.fit(X_train, y_train)

# Предсказание тестовых данных
y_test_pred = classifier.predict(X_test)

# Вычисление F-меры с помощью кросс-валидации
f1 = model_selection.cross_val_score(
    classifier,
    X,
    y,
    scoring='f1_weighted',
    cv=3
)

print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Оценка точности и полноты
from sklearn.metrics import classification_report

print("\nClassification report:")
print(classification_report(y_test, y_test_pred))

# Три тестовые точки
input_data = [
    '28', 'Private', '150000', 'Bachelors', '13',
    'Never-married', 'Prof-specialty', 'Not-in-family',
    'White', 'Female', '0', '0', '40', 'United-States'
]

input_data_2 = [
    '40', 'Private', '80000', 'HS-grad', '9',
    'Married-civ-spouse', 'Craft-repair', 'Husband',
    'White', 'Male', '0', '0', '40', 'United-States'
]

input_data_3 = [
    '52', 'Self-emp-not-inc', '200000', 'Masters', '14',
    'Married-civ-spouse', 'Exec-managerial', 'Husband',
    'White', 'Male', '5000', '0', '50', 'United-States'
]

# Функция кодирования и предсказания
def predict_income(input_data):
    input_data_encoded = []
    count = 0

    for item in input_data:
        if item.isdigit():
            input_data_encoded.append(int(item))
        else:
            encoded_value = label_encoder[count].transform([item])[0]
            input_data_encoded.append(encoded_value)
            count += 1

    input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

    predicted_class = classifier.predict(input_data_encoded)

    return label_encoder[-1].inverse_transform(predicted_class)[0]


# Проверка всех трёх точек
print("\nPredictions:")

print("Точка 1:", predict_income(input_data))
print("Точка 2:", predict_income(input_data_2))
print("Точка 3:", predict_income(input_data_3))
