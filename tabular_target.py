import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from art.estimators.classification import SklearnClassifier


# Available datasets with descriptions
DATASETS = {
    'iris': {
        'loader': load_iris,
        'description': '3 classes, 4 features (sepal length/width, petal length/width)',
        'n_features': 4,
        'n_classes': 3
    },
    'breast_cancer': {
        'loader': load_breast_cancer,
        'description': '2 classes, 30 features (tumor characteristics)',
        'n_features': 30,
        'n_classes': 2
    },
    'diabetes': {
        'loader': load_diabetes,
        'description': 'Regression converted to 2 classes (binned), 10 features',
        'n_features': 10,
        'n_classes': 2
    }
}


def load_tabular_target(dataset_name='iris', model_type='mlp', test_size=0.3, random_state=42):
    """
    Load a tabular dataset and wrap a classifier in ART's SklearnClassifier.
    
    Args:
        dataset_name: 'iris', 'breast_cancer', or 'diabetes'
        model_type: 'mlp', 'random_forest', 'gradient_boost', or 'logistic'
        test_size: proportion for test set
        random_state: for reproducibility
    
    Returns:
        art_classifier: ART-wrapped classifier
        x_test: test features (numpy array)
        y_test: test labels (numpy array)
        x_train: train features (for reference)
        y_train: train labels (for reference)
    """
    
    # Load dataset
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Choose from: {list(DATASETS.keys())}")
    
    data_info = DATASETS[dataset_name]
    data = data_info['loader']()
    
    X = data.data
    y = data.target
    
    # For diabetes (regression), convert to binary classification
    if dataset_name == 'diabetes':
        median_y = np.median(y)
        y = (y > median_y).astype(int)
        data_info['n_classes'] = 2
    
    print(f"[TabularTarget] Dataset: {dataset_name}")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Features: {data_info['n_features']}")
    print(f"  - Classes: {data_info['n_classes']}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Select and train model
    if model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=500,
            random_state=random_state,
            early_stopping=True
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
    elif model_type == 'gradient_boost':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            max_iter=500,
            random_state=random_state
        )
    else:
        raise ValueError(f"Model type {model_type} not found. Choose from: mlp, random_forest, gradient_boost, logistic")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate clean accuracy
    y_pred = model.predict(X_test)
    clean_acc = accuracy_score(y_test, y_pred)
    
    print(f"  - Model: {model_type}")
    print(f"  - Clean test accuracy: {clean_acc*100:.1f}%")
    
    # Wrap with ART
    # For tabular data, we need to provide feature bounds
    # Estimate bounds from training data
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    
    # Add small margin for numerical stability
    clip_values = (x_min - 0.1, x_max + 0.1)
    
    art_classifier = SklearnClassifier(
        model=model,
        clip_values=clip_values
    )
    
    return art_classifier, X_test.astype(np.float32), y_test.astype(np.int64), X_train, y_train


if __name__ == "__main__":
    # Test each dataset
    for dataset in ['iris', 'breast_cancer']:
        print(f"\n{'='*50}")
        classifier, x_test, y_test, _, _ = load_tabular_target(dataset, model_type='mlp')
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")