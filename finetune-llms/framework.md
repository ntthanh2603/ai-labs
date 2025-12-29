# Hướng dẫn Code: PyTorch, TensorFlow, Scikit-learn

## 1. Scikit-learn (Classical ML)

### Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
```

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

# Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2: {r2_score(y_test, y_pred)}")
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

### SVM

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
```

### XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
```

### Pipeline với Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid = GridSearchCV(XGBClassifier(), params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
```

---

## 2. PyTorch (Deep Learning)

### Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
```

### Simple Neural Network

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Training Loop

```python
# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
```

### Custom Dataset

```python
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader
dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### CNN for Images

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### RNN/LSTM for Sequences

```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```

### Save & Load Model

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Transfer Learning (ResNet)

```python
import torchvision.models as models

# Load pretrained ResNet
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Only train last layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### GPU Training

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# In training loop
data = data.to(device)
targets = targets.to(device)
```

---

## 3. TensorFlow/Keras (Deep Learning)

### Setup

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

### Simple Neural Network

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

### CNN for Images

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### RNN/LSTM for Sequences

```python
model = models.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Functional API (Complex Models)

```python
# Input
inputs = keras.Input(shape=(784,))

# Branches
x1 = layers.Dense(64, activation='relu')(inputs)
x1 = layers.Dense(32, activation='relu')(x1)

x2 = layers.Dense(64, activation='relu')(inputs)
x2 = layers.Dense(32, activation='relu')(x2)

# Concatenate
concat = layers.Concatenate()([x1, x2])

# Output
outputs = layers.Dense(10, activation='softmax')(concat)

model = keras.Model(inputs=inputs, outputs=outputs)
```

### Custom Training Loop

```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
```

### Callbacks

```python
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )
]

model.fit(X_train, y_train, callbacks=callbacks, epochs=50)
```

### Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val)
)
```

### Transfer Learning (VGG16)

```python
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### Save & Load Model

```python
# Save entire model
model.save('my_model.h5')

# Load model
model = keras.models.load_model('my_model.h5')

# Save only weights
model.save_weights('weights.h5')
model.load_weights('weights.h5')
```

---

## Comparison Cheat Sheet

| Feature               | Scikit-learn | PyTorch                 | TensorFlow/Keras              |
| --------------------- | ------------ | ----------------------- | ----------------------------- |
| **Best for**          | Classical ML | Research, custom models | Production, quick prototyping |
| **Ease of use**       | ⭐⭐⭐⭐⭐   | ⭐⭐⭐                  | ⭐⭐⭐⭐                      |
| **Flexibility**       | ⭐⭐         | ⭐⭐⭐⭐⭐              | ⭐⭐⭐⭐                      |
| **Community**         | Large        | Very Large              | Very Large                    |
| **GPU support**       | No           | Yes                     | Yes                           |
| **Mobile deployment** | No           | Yes (limited)           | Yes (TFLite)                  |

## Quick Tips

### Scikit-learn

- Tốt cho: Tabular data, quick baselines
- Always use pipelines cho reproducibility
- Cross-validation cho small datasets
- Grid/Random search cho tuning

### PyTorch

- Tốt cho: Research, custom architectures
- More "Pythonic" và intuitive
- Better debugging với standard Python
- Use `.to(device)` cho GPU
- Remember `.eval()` khi inference

### TensorFlow/Keras

- Tốt cho: Production deployment
- Keras API rất dễ dùng
- TensorBoard cho visualization
- TFLite cho mobile
- SavedModel format cho serving

## Common Workflow

```
1. Load Data
   ├─ Scikit-learn: pandas/numpy
   ├─ PyTorch: Dataset + DataLoader
   └─ TensorFlow: tf.data.Dataset

2. Preprocess
   ├─ Scikit-learn: StandardScaler, LabelEncoder
   ├─ PyTorch: torchvision.transforms
   └─ TensorFlow: tf.keras.preprocessing

3. Build Model
   ├─ Scikit-learn: model = RandomForest()
   ├─ PyTorch: class Model(nn.Module)
   └─ TensorFlow: keras.Sequential()

4. Train
   ├─ Scikit-learn: model.fit(X, y)
   ├─ PyTorch: Training loop với backward()
   └─ TensorFlow: model.fit(X, y)

5. Evaluate
   ├─ Scikit-learn: model.score() / metrics
   ├─ PyTorch: model.eval() + manual metrics
   └─ TensorFlow: model.evaluate()

6. Predict
   ├─ Scikit-learn: model.predict()
   ├─ PyTorch: model(tensor)
   └─ TensorFlow: model.predict()
```

---

**Lưu ý:** Đây là hướng dẫn cơ bản, mỗi framework có rất nhiều advanced features khác!
