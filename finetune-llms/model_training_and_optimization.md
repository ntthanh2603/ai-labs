# Model Training & Optimization

## Giới thiệu

Model Training là quá trình học từ dữ liệu để mô hình có thể dự đoán chính xác. Optimization là việc điều chỉnh để model hoạt động tốt nhất.

## Quy trình Training

1. **Chuẩn bị dữ liệu**: Split train/validation/test, normalize, augmentation
2. **Khởi tạo model**: Chọn architecture và khởi tạo weights
3. **Forward pass**: Đưa data qua model để tính prediction
4. **Tính loss**: So sánh prediction với ground truth
5. **Backward pass**: Tính gradients qua backpropagation
6. **Update weights**: Dùng optimizer để cập nhật parameters
7. **Lặp lại**: Repeat cho đến khi converge

## Loss Functions

| Task                  | Loss Function             | Khi nào dùng             |
| --------------------- | ------------------------- | ------------------------ |
| Regression            | MSE, MAE, Huber           | Dự đoán giá trị liên tục |
| Binary Classification | Binary Cross-Entropy      | Phân loại 2 classes      |
| Multi-class           | Categorical Cross-Entropy | Phân loại nhiều classes  |
| Object Detection      | IoU Loss                  | Bounding box regression  |

## Optimization Algorithms

### Gradient Descent Variants

**SGD (Stochastic Gradient Descent)**

- Cập nhật weights sau mỗi sample
- Nhanh nhưng không ổn định

**Mini-batch GD**

- Cập nhật sau mỗi batch (thường 32-256 samples)
- Cân bằng giữa tốc độ và độ ổn định

**Batch GD**

- Cập nhật sau khi xem toàn bộ dataset
- Chậm nhưng rất ổn định

### Advanced Optimizers

**Adam** (Most popular)

- Tự động điều chỉnh learning rate cho từng parameter
- Kết hợp momentum và adaptive learning rate
- Tốt cho hầu hết bài toán

**RMSprop**

- Tốt cho RNN và sequential data
- Giải quyết vấn đề vanishing/exploding gradients

**AdaGrad**

- Tốt cho sparse data
- Learning rate tự động giảm dần theo thời gian

**SGD with Momentum**

- Tích lũy gradient của các steps trước
- Giúp vượt qua local minima

## Hyperparameters chính

### 1. Learning Rate

- **Quá lớn**: Model không converge, loss oscillates
- **Quá nhỏ**: Training rất chậm, có thể stuck ở local minima
- **Optimal**: Thường 0.001, 0.01, 0.1
- **Tip**: Dùng learning rate scheduler để giảm dần

### 2. Batch Size

- **Nhỏ (16-64)**: Noisy gradients, generalizes tốt, ít memory
- **Lớn (256-1024)**: Stable gradients, overfits dễ, tốn memory
- **Trade-off**: Batch lớn = training nhanh nhưng có thể generalize kém

### 3. Number of Epochs

- Số lần model xem toàn bộ training data
- Quá ít: Underfitting
- Quá nhiều: Overfitting
- Solution: Early stopping

### 4. Model Architecture

- Số layers, neurons per layer
- Activation functions
- Deeper ≠ Better (có thể overfit)

## Regularization Techniques

### 1. L1 & L2 Regularization

- **L1 (Lasso)**: Đẩy weights về 0, làm feature selection
- **L2 (Ridge)**: Giữ weights nhỏ, smooth model
- **Elastic Net**: Kết hợp L1 và L2

### 2. Dropout

- Randomly "tắt" một phần neurons trong training
- Ngăn neurons phụ thuộc lẫn nhau
- Typical rate: 0.2 - 0.5

### 3. Early Stopping

- Dừng training khi validation loss không giảm nữa
- Ngăn overfitting một cách tự nhiên

### 4. Data Augmentation

- Tăng đa dạng của training data
- **Image**: Rotate, flip, crop, color jitter
- **Text**: Synonym replacement, back-translation
- **Audio**: Noise, speed change, pitch shift

### 5. Batch Normalization

- Normalize activations giữa các layers
- Stabilize và speed up training
- Có tác dụng regularization nhẹ

## Hyperparameter Tuning Methods

### 1. Manual Tuning

- Dựa vào kinh nghiệm và intuition
- Tốn thời gian nhưng hiểu model sâu

### 2. Grid Search

- Thử tất cả combinations trong grid
- Chắc chắn tìm được best trong grid
- Chậm và expensive

### 3. Random Search

- Random sampling trong hyperparameter space
- Nhanh hơn grid search
- Thường cho kết quả tương đương

### 4. Bayesian Optimization

- Học từ previous trials để chọn next hyperparameters
- Thông minh và efficient nhất
- Tools: Optuna, Hyperopt, Ray Tune

### 5. AutoML

- Tự động tìm architecture và hyperparameters
- Tools: Auto-sklearn, TPOT, Google AutoML

## Training Strategies

### 1. Transfer Learning

- Sử dụng pre-trained model
- Fine-tune trên data mới
- Tiết kiệm thời gian và data

### 2. Curriculum Learning

- Train từ dễ đến khó
- Model học ổn định hơn

### 3. Multi-task Learning

- Train nhiều tasks cùng lúc
- Share representations giữa tasks

### 4. Progressive Training

- Tăng dần độ phức tạp của model hoặc data
- Ví dụ: Progressive GAN

## Common Problems & Solutions

### Overfitting

- **Dấu hiệu**: Train loss thấp, validation loss cao
- **Giải pháp**: Regularization, more data, simpler model, early stopping

### Underfitting

- **Dấu hiệu**: Cả train và validation loss đều cao
- **Giải pháp**: Complex model, more features, train longer, reduce regularization

### Vanishing Gradients

- **Dấu hiệu**: Gradients rất nhỏ, model không học
- **Giải pháp**: ReLU activation, batch normalization, skip connections, LSTM/GRU

### Exploding Gradients

- **Dấu hiệu**: Loss = NaN, weights = infinity
- **Giải pháp**: Gradient clipping, lower learning rate, batch normalization

### Slow Convergence

- **Dấu hiệu**: Loss giảm rất chậm
- **Giải pháp**: Higher learning rate, better optimizer (Adam), normalize data

### Class Imbalance

- **Giải pháp**: Class weights, oversampling, undersampling, SMOTE, focal loss

## Learning Rate Scheduling

### 1. Step Decay

- Giảm learning rate sau mỗi N epochs
- Ví dụ: Giảm 50% sau mỗi 10 epochs

### 2. Exponential Decay

- Learning rate giảm exponentially theo thời gian

### 3. Cosine Annealing

- Learning rate follows cosine curve
- Smooth decay với periodic restarts

### 4. ReduceLROnPlateau

- Giảm learning rate khi validation loss không giảm
- Adaptive và practical

### 5. Warm-up + Decay

- Tăng dần learning rate ở đầu training
- Sau đó decay theo schedule

## Evaluation Metrics

### Classification

- **Accuracy**: Tỉ lệ dự đoán đúng
- **Precision**: Trong các dự đoán positive, bao nhiêu thực sự positive
- **Recall**: Trong các positive thực tế, bao nhiêu được tìm thấy
- **F1-Score**: Harmonic mean của precision và recall
- **AUC-ROC**: Khả năng phân biệt classes

### Regression

- **MAE**: Mean Absolute Error - dễ interpret
- **MSE**: Mean Squared Error - penalize outliers nặng hơn
- **RMSE**: Root MSE - cùng unit với target
- **R²**: Tỉ lệ variance được giải thích bởi model

## Best Practices

1. **Luôn dùng validation set** - Không đánh giá trên training data
2. **Monitor metrics trong training** - Loss, accuracy, learning rate
3. **Save checkpoints** - Lưu best model theo validation metric
4. **Reproducibility** - Set random seeds
5. **Start simple** - Baseline model trước, rồi mới complex
6. **Visualize training** - Plot loss curves, confusion matrix
7. **Log experiments** - Sử dụng MLflow, Weights & Biases
8. **Cross-validation** - Đánh giá robust hơn trên small datasets
9. **Test trên unseen data** - Test set chỉ dùng 1 lần cuối cùng

## Tools & Frameworks

**Training Frameworks:**

- PyTorch, TensorFlow, Keras
- Scikit-learn (classical ML)
- XGBoost, LightGBM (gradient boosting)

**Experiment Tracking:**

- Weights & Biases
- MLflow
- TensorBoard

**Hyperparameter Tuning:**

- Optuna
- Ray Tune
- Hyperopt

**AutoML:**

- Auto-sklearn
- H2O AutoML
- Google Vertex AI

## Kết luận

Model Training & Optimization là nghệ thuật cân bằng giữa:

- **Bias vs Variance**: Simple vs complex model
- **Speed vs Accuracy**: Fast training vs best performance
- **Memory vs Performance**: Model size vs accuracy

Key takeaways:
✅ Bắt đầu với baseline đơn giản
✅ Monitor training process liên tục
✅ Regularization để tránh overfitting
✅ Tune hyperparameters systematically
✅ Evaluate trên unseen data

Training tốt = Data quality + Architecture phù hợp + Hyperparameters optimal!
