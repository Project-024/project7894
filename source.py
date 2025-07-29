import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Config
emotion_labels = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]
data_dir = r"D:\MAHESH\facial emotion\mahesh\archive\train"
sequence_length = 5
img_size = (224, 224)
batch_size = 8
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
label_map = {label: idx for idx, label in enumerate(emotion_labels)}
num_classes = len(emotion_labels)

# Feature Extractor
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1]
feature_extractor = nn.Sequential(*modules).to(device)
feature_extractor.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Data Preparation
def extract_features_from_sequence(image_files):
    sequence = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = feature_extractor(img_tensor)
        sequence.append(feature.view(-1).cpu().numpy())
    return np.array(sequence)

X_data, y_data = [], []

for label in emotion_labels:
    folder = os.path.join(data_dir, label)
    image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i in range(0, len(image_files) - sequence_length + 1, sequence_length):
        seq_imgs = image_files[i:i + sequence_length]
        seq_feat = extract_features_from_sequence(seq_imgs)
        X_data.append(seq_feat)
        y_data.append(label_map[label])

X_data = np.array(X_data)
y_data = np.array(y_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data, random_state=42)

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = EmotionDataset(X_train, y_train)
test_dataset = EmotionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# LSTM Classifier
class EmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.relu(self.fc1(hn[-1]))
        return self.fc2(out)

input_size = X_data.shape[2]
model = EmotionLSTM(input_size=input_size, hidden_size=128, num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    train_loss = total_loss / len(train_dataset)
    train_acc = correct / len(train_dataset)

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    test_loss = val_loss / len(test_dataset)
    test_acc = val_correct / len(test_dataset)

    print(f"📘 Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% || "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

# Save model
torch.save(model.state_dict(), "emotion_lstm_model.pt")
print("\n✅ PyTorch model saved as 'emotion_lstm_model.pt'")
