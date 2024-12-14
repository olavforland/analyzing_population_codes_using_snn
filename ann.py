import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import safe_sparse_dot

SEED = 42

def compute_mlp_activations(response, y, n_neurons):
    clf = MLPClassifier(hidden_layer_sizes=(n_neurons,), max_iter=1000, random_state=SEED)
    # Define and train MLPClassifier
    clf.fit(response, y)

    # Compute activations for each layer
    activations = []
    X = response.copy()

    # Iterate through the layers in the MLP
    for i, (coefs, intercepts) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        # Compute layer's activation
        X = safe_sparse_dot(X, coefs) + intercepts  # Linear transformation
        if i != len(clf.coefs_) - 1:  # Apply activation function (ReLU for hidden layers)
            X = np.maximum(X, 0)
        activations.append(X)  # Store activations

    return np.array(activations[-2]) 

from sklearn.svm import SVC


def compute_margin(X, y):
    # Train an SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=SEED)
    svm_classifier.fit(X, y)

    # Retrieve the margin (1 / ||w||, where w is the weight vector)
    w = svm_classifier.coef_[0]  # Weight vector
    margin = 1 / np.linalg.norm(w)  # Margin

    return margin