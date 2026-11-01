import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from PIL import Image
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



#НАСТРОЙКИ
TRAIN_DIR = "dataset/seg_train/seg_train"
TEST_DIR = "dataset/seg_test/seg_test"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 5

print("Устройство:", DEVICE)



#ПОДГОТОВКА ДАННЫХ
weights = models.ResNet18_Weights.DEFAULT

transform = weights.transforms()

def load_images(folder):
    images = []
    labels = []

    classes = sorted([
        name for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ])

    class_to_idx = {
        name: index for index, name in enumerate(classes)
    }

    for class_name in classes:
        class_path = os.path.join(folder, class_name)

        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)

            try:
                image = Image.open(path).convert("RGB")
                images.append(transform(image))
                labels.append(class_to_idx[class_name])
            except Exception:
                continue

    return torch.stack(images), np.array(labels), classes


print("Загрузка обучающих изображений...")
X_train_images, y_train, classes = load_images(TRAIN_DIR)

print("Загрузка тестовых изображений...")
X_test_images, y_test, test_classes = load_images(TEST_DIR)

assert classes == test_classes, "Порядок классов не совпадает!"

print("Классы:", classes)
print("Обучающая выборка:", len(y_train))
print("Тестовая выборка:", len(y_test))


#ЗАГРУЗКА RESNET18
model = models.resnet18(weights=weights)

#Удаляем последний классификационный слой
model.fc = nn.Identity()

model = model.to(DEVICE)
model.eval()


#ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
def extract_features(images):
    features = []

    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i + BATCH_SIZE].to(DEVICE)

            output = model(batch)

            features.append(output.cpu().numpy())

    return np.concatenate(features)


print("Извлечение признаков...")

X_train = extract_features(X_train_images)
X_test = extract_features(X_test_images)

print("Размер признаков:", X_train.shape)


#FEATURE EXTRACTION
print("\nFEATURE EXTRACTION")

feature_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42
    )
)

feature_model.fit(X_train, y_train)

y_pred_feature = feature_model.predict(X_test)


#ОЦЕНКА FEATURE EXTRACTION
def evaluate_model(y_true, y_pred, name):
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"\n{name}")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=classes,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title(name)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    plt.savefig(name.replace(" ", "_") + ".png")
    plt.show()

    return accuracy, precision, recall


results_feature = evaluate_model(
    y_test,
    y_pred_feature,
    "Feature Extraction"
)


#FINE-TUNING
print("\nFINE-TUNING")

# Создаём новую ResNet18
fine_model = models.resnet18(weights=weights)

# Замораживаем все слои
for param in fine_model.parameters():
    param.requires_grad = False

# Размораживаем последний residual-блок
for param in fine_model.layer4.parameters():
    param.requires_grad = True

# Заменяем классификатор
fine_model.fc = nn.Linear(
    fine_model.fc.in_features,
    len(classes)
)

fine_model = fine_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, fine_model.parameters()),
    lr=0.0001
)


#ПОДГОТОВКА ДАТАЛОАДЕРОВ
from torch.utils.data import TensorDataset, DataLoader

train_tensor_dataset = TensorDataset(
    X_train_images,
    torch.tensor(y_train, dtype=torch.long)
)

test_tensor_dataset = TensorDataset(
    X_test_images,
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(
    train_tensor_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_tensor_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


#ОБУЧЕНИЕ FINE-TUNING
for epoch in range(EPOCHS):

    fine_model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = fine_model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"Loss: {running_loss / len(train_loader):.4f}"
    )


#ПРЕДСКАЗАНИЕ FINE-TUNING
fine_model.eval()

predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = fine_model(images)

        predicted = torch.argmax(outputs, dim=1)

        predictions.extend(predicted.cpu().numpy())


y_pred_fine = np.array(predictions)


#ОЦЕНКА FINE-TUNING
results_fine = evaluate_model(
    y_test,
    y_pred_fine,
    "Fine-tuning"
)


#СРАВНЕНИЕ РЕЗУЛЬТАТОВ
print("\nСРАВНЕНИЕ МОДЕЛЕЙ")

print(
    f"{'Метрика':<15}"
    f"{'Feature Extraction':<22}"
    f"{'Fine-tuning':<15}"
)

metrics = ["Accuracy", "Precision", "Recall"]

for i, metric in enumerate(metrics):

    print(
        f"{metric:<15}"
        f"{results_feature[i]:<22.4f}"
        f"{results_fine[i]:<15.4f}"
    )
