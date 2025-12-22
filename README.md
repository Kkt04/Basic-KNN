# K-Nearest Neighbors (KNN) Algorithm - Iris Dataset

## Overview
This project demonstrates the implementation and analysis of the K-Nearest Neighbors (KNN) algorithm using Python's Scikit-Learn library on the classic Iris dataset.

## Dataset
The Iris dataset contains 150 samples with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Target classes: 3 species of Iris flowers (Setosa, Versicolor, Virginica)

## Features

### 1. Data Exploration
- Loading and exploring the Iris dataset
- Visualizing feature relationships using scatter plots
- Examining data distribution and target classes

### 2. Data Preprocessing
- Train-test split (80-20 ratio)
- Feature scaling demonstration using StandardScaler

### 3. Model Implementation
- KNN classifier with customizable k values
- Decision boundary visualization using mlxtend
- Model training and prediction

### 4. Model Evaluation
- Testing different k values (1-50)
- Accuracy score calculation
- Performance visualization across different k values

### 5. Distance Metrics
- Euclidean distance calculation implementation
- Comparison of distances before and after feature scaling

## Key Concepts Demonstrated

### KNN Algorithm
The K-Nearest Neighbors algorithm works by:
1. Finding the 'k' nearest data points to a given input
2. Predicting class/value based on majority class or average of neighbors

### Important Considerations
- **Feature Scaling**: Demonstrates why scaling is crucial for KNN
- **Optimal k Selection**: Analysis shows why smaller k values may give higher accuracy
- **Distance Metrics**: Implementation of Euclidean distance calculation

## Requirements
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
import numpy as np
```

## Installation
```bash
pip install pandas matplotlib seaborn scikit-learn mlxtend numpy
```

## Usage

### Basic KNN Implementation
```python
# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

### Finding Optimal k
The notebook includes code to test k values from 1 to 50 and plot accuracy scores to find the optimal k value.

## Visualizations
1. **Scatter Plot**: 2D visualization of Iris features colored by species
2. **Decision Boundary**: Visual representation of KNN classification regions
3. **Accuracy vs k Plot**: Performance comparison across different k values

## Key Insights

### Why Small k Values May Give Higher Accuracy
- Lower k values make the model more sensitive to local patterns
- Can lead to overfitting on training data
- Trade-off between bias and variance

### Feature Scaling Impact
The notebook demonstrates how feature scaling affects distance calculations:
- Without scaling: Features with larger scales dominate distance calculations
- With scaling: All features contribute proportionally to distance metrics

## File Structure
```
├── Basic_KNN.ipynb          # Main Jupyter notebook
└── README.md               # This file
```

## Results
The model demonstrates:
- High accuracy on the Iris dataset
- Clear decision boundaries between classes
- Importance of hyperparameter tuning (k selection)
- Impact of feature scaling on model performance

## Future Enhancements
- Cross-validation for more robust k selection
- Comparison with other distance metrics (Manhattan, Minkowski)
- Feature importance analysis
- Extension to other datasets

## References
- [Scikit-Learn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- Iris Dataset: Fisher, R.A. "The use of multiple measurements in taxonomic problems" (1936)

## License
This project is available for educational purposes.

## Contributing
Feel free to fork this repository and submit pull requests for any improvements.

## Contact
For questions or feedback, please open an issue in the repository.

---

