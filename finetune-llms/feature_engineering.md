# Feature Engineering - Kỹ Thuật Tạo Đặc Trưng

## Giới thiệu

Feature Engineering là quá trình biến đổi dữ liệu thô thành các đặc trưng (features) có ý nghĩa giúp mô hình machine learning học tốt hơn. Đây được coi là một trong những bước quan trọng nhất trong pipeline machine learning, có thể quyết định thành công hay thất bại của một dự án.

## Tại sao Feature Engineering quan trọng?

Feature Engineering quan trọng vì:

- Cải thiện độ chính xác của mô hình đáng kể
- Giảm độ phức tạp của mô hình
- Tăng tốc độ training và inference
- Giúp mô hình hiểu được mối quan hệ ẩn trong dữ liệu
- Có thể quan trọng hơn cả việc chọn thuật toán

## Các kỹ thuật Feature Engineering cơ bản

### 1. Xử lý Missing Values (Giá trị thiếu)

**Mean/Median/Mode Imputation**

- Thay thế giá trị thiếu bằng mean (số liên tục), median (có outliers) hoặc mode (categorical)
- Đơn giản nhưng có thể làm mất thông tin về sự phân bố

```python
df['age'].fillna(df['age'].mean(), inplace=True)
df['category'].fillna(df['category'].mode()[0], inplace=True)
```

**Forward Fill / Backward Fill**

- Sử dụng cho time series data
- Forward fill: dùng giá trị trước đó
- Backward fill: dùng giá trị sau đó

**KNN Imputation**

- Sử dụng k-nearest neighbors để dự đoán giá trị thiếu
- Tốt hơn mean/median nhưng tốn thời gian hơn

**Đánh dấu Missing Values**

- Tạo thêm binary feature để đánh dấu vị trí có giá trị thiếu
- Đôi khi việc thiếu dữ liệu cũng là một thông tin quan trọng

### 2. Encoding Categorical Variables

**Label Encoding**

- Chuyển đổi categories thành số (0, 1, 2, ...)
- Phù hợp với ordinal data (có thứ tự: small, medium, large)

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['size'] = le.fit_transform(df['size'])
```

**One-Hot Encoding**

- Tạo binary column cho mỗi category
- Phù hợp với nominal data (không có thứ tự: color, country)
- Có thể gây curse of dimensionality nếu có nhiều categories

```python
pd.get_dummies(df['color'], prefix='color')
```

**Target Encoding / Mean Encoding**

- Thay thế category bằng mean của target variable
- Mạnh mẽ nhưng dễ bị overfitting, cần validation set

**Frequency Encoding**

- Thay category bằng tần suất xuất hiện của nó
- Đơn giản và hiệu quả với high cardinality features

**Binary Encoding**

- Chuyển category thành binary representation
- Tiết kiệm không gian hơn one-hot encoding

### 3. Feature Scaling

**Standardization (Z-score Normalization)**

- Chuyển data về mean = 0, std = 1
- Không bị ảnh hưởng bởi outliers nhiều
- Công thức: (x - μ) / σ

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

**Min-Max Normalization**

- Scale data về khoảng [0, 1] hoặc [-1, 1]
- Bị ảnh hưởng bởi outliers
- Công thức: (x - min) / (max - min)

**Robust Scaling**

- Sử dụng median và IQR thay vì mean và std
- Tốt hơn khi có outliers

**Log Transformation**

- Áp dụng log để giảm skewness
- Hữu ích cho right-skewed distributions

### 4. Feature Creation

**Polynomial Features**

- Tạo các đặc trưng bậc cao (x², x³, x\*y, ...)
- Giúp capture non-linear relationships

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

**Interaction Features**

- Kết hợp nhiều features với nhau
- Ví dụ: BMI = weight / (height²)

**Domain-Specific Features**

- Tạo features dựa trên hiểu biết về domain
- Ví dụ: từ datetime tạo ra day_of_week, is_weekend, hour, season

**Aggregation Features**

- Sum, mean, max, min, std của nhóm features
- Ví dụ: total_spending, average_transaction_value

**Text Features**

- TF-IDF, Bag of Words, Word Embeddings
- N-grams, character features

### 5. Binning / Discretization

**Equal-Width Binning**

- Chia thành các bins có width bằng nhau
- Đơn giản nhưng có thể không cân bằng về số lượng samples

**Equal-Frequency Binning (Quantile)**

- Mỗi bin có số lượng samples gần bằng nhau
- Tốt hơn khi distribution không đều

**Custom Binning**

- Chia bins dựa trên domain knowledge
- Ví dụ: age groups (0-18, 19-35, 36-60, 60+)

### 6. Feature Extraction

**Principal Component Analysis (PCA)**

- Giảm số chiều bằng cách tạo ra các principal components
- Giữ lại phần lớn variance của data

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
```

**Linear Discriminant Analysis (LDA)**

- Supervised dimensionality reduction
- Tối đa hóa sự phân tách giữa các classes

**t-SNE và UMAP**

- Visualization và dimensionality reduction
- Tốt cho non-linear relationships

**Autoencoders**

- Deep learning approach cho feature extraction
- Có thể học được complex representations

## Kỹ thuật nâng cao

### 1. Feature Selection

**Filter Methods**

- Correlation coefficient
- Chi-square test
- Mutual information
- ANOVA F-test

**Wrapper Methods**

- Forward selection
- Backward elimination
- Recursive Feature Elimination (RFE)

**Embedded Methods**

- Lasso (L1 regularization)
- Tree-based feature importance
- Elastic Net

### 2. Time Series Features

**Lag Features**

- Giá trị tại thời điểm trước đó (t-1, t-2, ...)

**Rolling Window Statistics**

- Moving average, rolling std, rolling max/min

**Date-Time Features**

- Year, month, day, hour, minute
- Day of week, day of year
- Is weekend, is holiday
- Season, quarter

**Trend và Seasonality**

- Decompose time series thành trend, seasonal, residual components

### 3. Image Features

**Color Features**

- Color histograms, color moments
- Dominant colors

**Texture Features**

- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)
- Gabor filters

**Shape Features**

- Edge detection
- Contours, moments

**Deep Learning Features**

- Transfer learning từ pre-trained CNNs
- Feature maps từ intermediate layers

### 4. Text Features

**Statistical Features**

- Document length, word count
- Average word length
- Punctuation count

**Linguistic Features**

- POS tags frequency
- Named entities
- Sentiment scores

**Advanced Embeddings**

- Word2Vec, GloVe
- BERT, GPT embeddings
- Doc2Vec cho document-level

## Best Practices

### 1. Feature Engineering Workflow

1. **Hiểu dữ liệu**: EDA (Exploratory Data Analysis) kỹ lưỡng
2. **Brainstorm features**: Dựa trên domain knowledge
3. **Tạo features**: Implement các ideas
4. **Đánh giá features**: Test trên validation set
5. **Iterate**: Refine và tạo thêm features mới
6. **Feature selection**: Loại bỏ features không hữu ích

### 2. Tránh Data Leakage

Data leakage xảy ra khi information từ test set "leak" vào training process:

- Fit scaler/encoder chỉ trên training set
- Tạo features dựa trên toàn bộ dataset trước khi split
- Sử dụng target variable để tạo features

**Cách tránh:**

- Luôn split data trước
- Sử dụng Pipeline hoặc ColumnTransformer
- Cross-validation đúng cách

### 3. Feature Documentation

Luôn document các features:

- Tên feature và ý nghĩa
- Công thức tính toán
- Data type và range
- Missing value handling
- Business logic đằng sau

### 4. Automation

**Feature Store**

- Centralized repository cho features
- Đảm bảo consistency giữa training và production

**AutoML Tools**

- Featuretools: Automated feature engineering
- TPOT: Genetic programming approach
- H2O AutoML: End-to-end automation

## Tools và Libraries

**Python Libraries:**

- **scikit-learn**: StandardScaler, LabelEncoder, PolynomialFeatures
- **pandas**: Data manipulation, datetime features
- **featuretools**: Automated feature engineering
- **category_encoders**: Advanced categorical encoding
- **feature-engine**: Feature engineering pipelines
- **tsfresh**: Time series feature extraction

**Visualization:**

- **matplotlib, seaborn**: Phân tích features
- **plotly**: Interactive plots
- **pandas-profiling**: Automated EDA

## Ví dụ thực tế

### Dự án House Price Prediction

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv('house_data.csv')

# 1. Missing values
df['lot_frontage'].fillna(df['lot_frontage'].median(), inplace=True)

# 2. Date features
df['year_sold'] = pd.to_datetime(df['sale_date']).dt.year
df['month_sold'] = pd.to_datetime(df['sale_date']).dt.month
df['age'] = df['year_sold'] - df['year_built']

# 3. Interaction features
df['total_sf'] = df['first_floor_sf'] + df['second_floor_sf']
df['total_bathrooms'] = df['full_bath'] + 0.5 * df['half_bath']

# 4. Polynomial features
df['living_area_squared'] = df['living_area'] ** 2

# 5. Binning
df['price_range'] = pd.cut(df['price'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])

# 6. Encoding
le = LabelEncoder()
df['neighborhood_encoded'] = le.fit_transform(df['neighborhood'])

# 7. Scaling
scaler = StandardScaler()
df[['living_area', 'lot_size']] = scaler.fit_transform(df[['living_area', 'lot_size']])
```

### Dự án Customer Churn Prediction

```python
# Aggregation features
customer_features = transactions.groupby('customer_id').agg({
    'transaction_amount': ['sum', 'mean', 'std', 'max', 'min'],
    'transaction_date': ['count'],
}).reset_index()

# Recency features
customer_features['days_since_last_purchase'] = (
    pd.to_datetime('today') - transactions.groupby('customer_id')['transaction_date'].max()
).dt.days

# Frequency features
customer_features['purchase_frequency'] = (
    transactions.groupby('customer_id')['transaction_date'].count() /
    customer_features['days_since_first_purchase']
)

# Monetary features
customer_features['avg_transaction_value'] = (
    transactions.groupby('customer_id')['transaction_amount'].mean()
)
```

## Challenges và Solutions

**Challenge 1: High Cardinality Categorical Variables**

- Solution: Target encoding, frequency encoding, embedding layers

**Challenge 2: Imbalanced Data**

- Solution: SMOTE, class weights, stratified sampling

**Challenge 3: Curse of Dimensionality**

- Solution: Feature selection, PCA, regularization

**Challenge 4: Non-stationary Time Series**

- Solution: Differencing, detrending, adaptive features

**Challenge 5: Computational Cost**

- Solution: Feature selection, parallel processing, sampling

## Kết luận

Feature Engineering là một nghệ thuật kết hợp giữa domain knowledge, creativity và technical skills. Không có công thức chung nào cho mọi bài toán - mỗi dataset và problem cần approach riêng. Thực hành nhiều và học hỏi từ các Kaggle competitions là cách tốt nhất để master kỹ năng này.

Key takeaways:

- Hiểu data và domain là nền tảng
- Thử nghiệm nhiều techniques khác nhau
- Validate features trên hold-out set
- Tránh data leakage bằng mọi giá
- Document và automate process
- Iterate và improve liên tục

Feature Engineering tốt có thể cải thiện model performance hơn việc tune hyperparameters hay thử nhiều algorithms phức tạp!
