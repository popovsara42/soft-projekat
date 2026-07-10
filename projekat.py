import torch
import pandas as pd
import numpy as np

from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#priprema podataka- filmova za klasifikaciju i svih zanrova koji se pojavljuju
dataset = load_dataset(
    "skvarre/movie_posters-100k",
    streaming=True
)

data = []

for i, item in enumerate(dataset["train"]):
    if i >= 1000:
        break

    if item["image"] is not None and item["genres"] is not None and len(item["genres"]) > 0:
        item["Genre"] = "|".join(g["name"] for g in item["genres"])
        data.append(item)

df = pd.DataFrame(data)

print("Kolone:", df.columns)
print("Broj primera:", len(df))


genres = sorted(set(
    genre
    for row in df["Genre"]
    for genre in row.split("|")
))

genre_to_idx = {g: i for i, g in enumerate(genres)}

print("Broj žanrova:", len(genres))
print(genres)


train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


#racunanje tezina klasa
def calculate_pos_weight(df, genres, genre_to_idx):
    positive_counts = torch.zeros(len(genres))

    for genre_row in df["Genre"]:
        for g in genre_row.split("|"):
            positive_counts[genre_to_idx[g]] += 1

    negative_counts = len(df) - positive_counts

    pos_weight = torch.sqrt(negative_counts / (positive_counts + 1e-6))
    pos_weight = torch.clamp(pos_weight, max=5)

    return pos_weight


pos_weight = calculate_pos_weight(train_df, genres, genre_to_idx).to(DEVICE)


#yolo
yolo = YOLO("yolov8s-world.pt")

custom_classes = [
    "person", "gun", "knife", "sword", "car", "motorcycle",
    "book", "animal", "fire", "explosion", "robot", "spaceship",
    "monster", "zombie", "castle", "horse"
]

yolo.set_classes(custom_classes)


def get_object_vector(image):
    result = yolo(image, verbose=False)[0]

    object_vector = torch.zeros(len(custom_classes))

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        if cls_id < len(custom_classes):
            object_vector[cls_id] = max(object_vector[cls_id], conf)

    max_value = object_vector.max()

    if max_value > 0:
        object_vector = object_vector / max_value

    return object_vector


#transformacije
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#dataset
class MoviePosterDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        original_image = self.df.iloc[idx]["image"].convert("RGB")

        image = self.transform(original_image)

        object_vector = get_object_vector(original_image)

        label = torch.zeros(len(genres))

        for g in self.df.iloc[idx]["Genre"].split("|"):
            label[genre_to_idx[g]] = 1

        return image, object_vector, label


train_loader = DataLoader(
    MoviePosterDataset(train_df, train_transform),
    batch_size=32,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    MoviePosterDataset(val_df, eval_transform),
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    MoviePosterDataset(test_df, eval_transform),
    batch_size=32,
    shuffle=False
)

#model: resnet50+yolo
class PosterGenreModel(nn.Module):
    def __init__(self, num_genres, num_objects):
        super().__init__()

        resnet = models.resnet50(weights="DEFAULT")

        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.visual_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.object_fc = nn.Sequential(
            nn.Linear(num_objects, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.final_network = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_genres)
        )
        self.final_networkk = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_genres)
        )

    def unfreeze_last_cnn_layers(self):
        for param in self.cnn[-2].parameters():
            param.requires_grad = True

    def forward(self, image, object_vector):
        visual_features = self.cnn(image)
        visual_features = visual_features.view(visual_features.size(0), -1)

        visual_features = self.visual_fc(visual_features)
        object_features = self.object_fc(object_vector)

        combined_features = torch.cat(
            [visual_features, object_features],
            dim=1
        )

        logits = self.final_network(combined_features)

        return logits


model = PosterGenreModel(
    num_genres=len(genres),
    num_objects=len(custom_classes)
).to(DEVICE)


#loss i optimizacija
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)


#evaluacija
def evaluate(model, loader, threshold=0.5):
    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0

    with torch.no_grad():
        for images, object_vectors, labels in loader:
            images = images.to(DEVICE)
            object_vectors = object_vectors.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, object_vectors)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= threshold).int()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_predictions)

    return {
        "loss": total_loss,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred)
    }


def find_best_threshold(model, loader):
    best_threshold = 0.5
    best_micro_f1 = 0

    for threshold in np.arange(0.30, 0.81, 0.05):
        metrics = evaluate(model, loader, threshold=threshold)

        if metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = metrics["micro_f1"]
            best_threshold = threshold

    return best_threshold, best_micro_f1


#trening
EPOCHS = 14
PATIENCE = 5
MIN_DELTA = 0.003

best_val_f1 = 0
epochs_without_improvement = 0
best_threshold = 0.5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    if epoch == 5:
        model.unfreeze_last_cnn_layers()

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

        print("Odmrznut je poslednji CNN blok za fine tuning.")

    for images, object_vectors, labels in train_loader:
        images = images.to(DEVICE)
        object_vectors = object_vectors.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images, object_vectors)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    current_threshold, _ = find_best_threshold(model, val_loader)
    val_metrics = evaluate(model, val_loader, threshold=current_threshold)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("Train loss:", total_loss)
    print("Best threshold:", current_threshold)
    print("Validation Micro F1:", val_metrics["micro_f1"])
    print("Validation Macro F1:", val_metrics["macro_f1"])
    print("Validation Precision:", val_metrics["precision"])
    print("Validation Recall:", val_metrics["recall"])
    print("Validation Hamming Loss:", val_metrics["hamming_loss"])
    print("-" * 40)

    improvement = val_metrics["micro_f1"] - best_val_f1

    if improvement > MIN_DELTA:
        best_val_f1 = val_metrics["micro_f1"]
        best_threshold = current_threshold
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "best_model.pt")
        print("Sačuvan novi najbolji model.")
    else:
        epochs_without_improvement += 1
        print(f"Nema značajnog napretka. Pomak: {improvement:.5f}")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping aktiviran.")
        break


#test
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))

test_metrics = evaluate(model, test_loader, threshold=best_threshold)

print("TEST REZULTATI")
print("Korišćeni threshold:", best_threshold)
print("Micro F1:", test_metrics["micro_f1"])
print("Macro F1:", test_metrics["macro_f1"])
print("Precision:", test_metrics["precision"])
print("Recall:", test_metrics["recall"])
print("Hamming Loss:", test_metrics["hamming_loss"])
