"""
Neural network architectures and training for Deep-PrAE.

This module implements the neural network classifiers used to approximate
rare-event set boundaries in the Deep-PrAE framework.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from typing import List, Tuple, Optional

# Set device and random seeds for reproducibility
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"  # Force CPU for consistency
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class NeuralNetworkClassifier(nn.Module):
    """
    Feedforward neural network classifier for rare-event detection.

    Architecture: Input → Linear → ReLU → ... → Linear → ReLU → Linear → Output
    Uses ReLU activations in all hidden layers, no activation in output layer.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 2):
        """
        Initialize neural network classifier.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions (e.g., [10] for one hidden layer,
                        [8, 8, 4, 2] for four hidden layers)
            output_dim: Output dimension (2 for binary classification)
        """
        super(NeuralNetworkClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Variable) -> Variable:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [N, input_dim]

        Returns:
            Output logits of shape [N, output_dim]
        """
        return self.network(x)

    def extract_params(self) -> dict:
        """
        Extract network parameters for optimization model.

        Returns:
            Dictionary containing weights and biases for each layer,
            formatted for use in Pyomo optimization model.
        """
        W = {}
        layer_idx = 0

        for module in self.network:
            if isinstance(module, nn.Linear):
                # Extract weight matrix and bias vector
                weight = module.state_dict()['weight'].cpu().numpy()
                bias = module.state_dict()['bias'].cpu().numpy()

                # Convert to nested dictionary format
                weight_dict = {
                    str(i): {str(j): float(weight[i][j]) for j in range(weight.shape[1])}
                    for i in range(weight.shape[0])
                }
                bias_dict = {str(i): float(bias[i]) for i in range(len(bias))}

                W[str(layer_idx)] = {
                    'weight': weight_dict,
                    'bias': bias_dict
                }
                layer_idx += 1

        return W


def train_classifier(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    hidden_dims: List[int],
    n_iters: int = 1000,
    batch_size: Optional[int] = None,
    lr: float = 5e-3,
    class_weights: List[float] = [1.0, 1.0],
    l2_reg: float = 0.0,
    log: bool = False,
    save_path: Optional[str] = None
) -> Tuple[NeuralNetworkClassifier, dict]:
    """
    Train neural network classifier for rare-event detection.

    Args:
        X_train: Training input of shape [N, D]
        Y_train: Training labels of shape [N,] (binary: 0 or 1)
        hidden_dims: List of hidden layer dimensions
        n_iters: Number of training iterations
        batch_size: Batch size (if None, uses n1/20 as per paper)
        lr: Learning rate
        class_weights: Weights for each class [non-rare, rare]
        l2_reg: L2 regularization coefficient
        log: Whether to print training progress
        save_path: Path to save trained model

    Returns:
        Tuple of (trained_model, training_history)
    """
    N, D = X_train.shape

    if batch_size is None:
        batch_size = max(1, N // 20)  # Default from paper

    # Create dataset
    train_dataset = []
    for i in range(N):
        train_dataset.append([X_train[i, :], Y_train[i]])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize network
    net = NeuralNetworkClassifier(
        input_dim=D,
        hidden_dims=hidden_dims,
        output_dim=2
    ).to(device)

    # Set up optimizer with L2 regularization
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2_reg)

    # Set up loss function with class weights
    class_weight = torch.Tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

    # Training loop
    num_epochs = int(np.ceil(n_iters / (len(train_dataset) / batch_size)))

    history = {
        'loss': [],
        'accuracy': [],
        'false_positive_rate': [],
        'false_negative_rate': []
    }

    net.train()
    iteration = 0

    for epoch in range(num_epochs):
        for batch_X, batch_Y in train_loader:
            iteration += 1

            X = Variable(torch.Tensor(batch_X.float())).to(device)
            Y = Variable(torch.Tensor(batch_Y.float())).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, Y.long())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute metrics
            if log:
                with torch.no_grad():
                    pred = outputs.max(1)[1]
                    label = Y.long()

                    accuracy = (pred == label).float().mean()
                    false_pos = torch.logical_and(pred == 1, label == 0).float().mean()
                    false_neg = torch.logical_and(pred == 0, label == 1).float().mean()

                    history['loss'].append(loss.item())
                    history['accuracy'].append(accuracy.item())
                    history['false_positive_rate'].append(false_pos.item())
                    history['false_negative_rate'].append(false_neg.item())

                    if iteration % 100 == 0:
                        print(f"Iter: {iteration}, Loss: {loss.item():.4f}, "
                              f"Acc: {accuracy.item():.3f}, "
                              f"FPR: {false_pos.item():.3f}, "
                              f"FNR: {false_neg.item():.3f}")

            if iteration >= n_iters:
                break

        if iteration >= n_iters:
            break

    # Save model if path provided
    if save_path is not None:
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    net.eval()
    return net, history


def tune_threshold(
    net: NeuralNetworkClassifier,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    method: str = 'bisection',
    target_fpr: float = 0.05
) -> float:
    """
    Tune classification threshold to obtain outer approximation.

    Args:
        net: Trained neural network
        X_val: Validation input
        Y_val: Validation labels
        method: Threshold tuning method ('bisection' or 'roc')
        target_fpr: Target false positive rate

    Returns:
        Optimal threshold value kappa
    """
    net.eval()

    with torch.no_grad():
        X_tensor = torch.Tensor(X_val).to(device)
        outputs = net(X_tensor)

        # Get prediction scores (difference between rare and non-rare logits)
        scores = outputs[:, 1] - outputs[:, 0]
        scores = scores.cpu().numpy()

    if method == 'bisection':
        # Binary search for threshold that gives outer approximation
        thresholds = np.sort(np.unique(scores))

        best_threshold = 0.0
        for threshold in thresholds:
            preds = (scores >= threshold).astype(float)

            # Check if it's an outer approximation (no false negatives)
            false_negs = np.logical_and(preds == 0, Y_val == 1).sum()

            if false_negs == 0:
                best_threshold = threshold
                break

        return best_threshold

    elif method == 'roc':
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(Y_val, scores)

        # Find threshold that achieves target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        return thresholds[idx]

    else:
        raise ValueError(f"Unknown method: {method}")
