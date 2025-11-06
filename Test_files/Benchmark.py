import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.io as sio
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import logging
import math
import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
#from SSM.utility import SimpleLSTM
from neural_ssm import DeepSSM, SSMConfig



# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 8590
    seed: int = 9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True

    # Early stopping
    patience: int = 50
    min_delta: float = 1e-6

    # Checkpointing
    save_dir: Path = Path("../checkpoints")
    save_best_only: bool = True


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    n_u: int = 1
    n_y: int = 1
    d_model: int = 16
    d_state: int = 11
    n_layers: int = 1
    ff: str = "LMLP"  # GLU | MLP | LMLP
    max_phase: float = math.pi / 50
    r_min: float = 0.7
    r_max: float = 0.98
    d_amp: int = 8
    param: str = 'l2ru'
    gamma: Optional[float] = 2
    init: str = 'rand'

    def to_ssm_config(self) -> SSMConfig:
        """Convert to SSMConfig object."""
        return SSMConfig(
            d_model=self.d_model,
            d_state=self.d_state,
            n_layers=self.n_layers,
            ff=self.ff,
            rmin=self.r_min,
            rmax=self.r_max,
            max_phase=self.max_phase,
            dim_amp=self.d_amp,
            param=self.param,
            gamma=self.gamma,
            init=self.init
        )


# ============================================================================
# Data Management
# ============================================================================

class SystemIDDataset(Dataset):
    """PyTorch Dataset for system identification data."""

    def __init__(self, u: torch.Tensor, y: torch.Tensor):
        """
        Args:
            u: Input tensor of shape (seq_len, n_u) or (batch, seq_len, n_u)
            y: Output tensor of shape (seq_len, n_y) or (batch, seq_len, n_y)
        """
        assert u.shape[0] == y.shape[0], "Input and output must have same length"
        self.u = u.float()
        self.y = y.float()

    def __len__(self) -> int:
        return len(self.u)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.u[idx], self.y[idx]


def load_mat_data(filepath: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load system identification data from MATLAB file.

    Args:
        filepath: Path to .mat file

    Returns:
        Tuple of (u_train, y_train, u_val, y_val) as PyTorch tensors
    """
    mat_data = sio.loadmat(filepath)

    u_train = torch.from_numpy(mat_data['uEst']).float()
    y_train = torch.from_numpy(mat_data['yEst']).float()
    u_val = torch.from_numpy(mat_data['uVal']).float()
    y_val = torch.from_numpy(mat_data['yVal']).float()

    return u_train, y_train, u_val, y_val


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-6):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Args:
            val_loss: Current validation loss
            model: Model to save if improved

        Returns:
            True if should stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0

        return self.early_stop

    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ============================================================================
# Training and Evaluation
# ============================================================================

class SystemIDTrainer:
    """Trainer for system identification models."""

    def __init__(
            self,
            model: nn.Module,
            train_config: TrainingConfig,
            criterion: Optional[nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Args:
            model: The model to train
            train_config: Training configuration
            criterion: Loss function (defaults to MSELoss)
            optimizer: Optimizer (defaults to Adam)
        """
        self.model = model
        self.config = train_config
        self.device = torch.device(train_config.device)
        self.model.to(self.device)

        # Set up loss and optimizer
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            model.parameters(), lr=train_config.learning_rate
        )

        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, u: torch.Tensor, y: torch.Tensor) -> float:
        """
        Train for one epoch.

        Args:
            u: Input tensor
            y: Target tensor

        Returns:
            Average training loss
        """
        self.model.train()

        # Move data to device
        u = u.to(self.device)
        y = y.to(self.device)

        # Forward pass
        y_pred, _ = self.model(u)
        y_pred = y_pred.squeeze()
        y = y.squeeze()

        # Compute loss
        loss = self.criterion(y_pred, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self, u: torch.Tensor, y: torch.Tensor, n: int = 50) -> float:
        """
        Validate the model.

        Args:
            n: Initialization window length
            u: Input tensor
            y: Target tensor

        Returns:
            Validation loss
        """
        self.model.eval()

        # Move data to device
        u = u.to(self.device)
        y = y.to(self.device)

        # Forward pass
        y_pred, _ = self.model(u)
        y_pred = y_pred.squeeze()
        y = y.squeeze()

        # Compute loss
        loss = self.criterion(y_pred[n:], y[n:])

        return loss.item()

    def fit(
            self,
            u_train: torch.Tensor,
            y_train: torch.Tensor,
            u_val: torch.Tensor,
            y_val: torch.Tensor,
            use_early_stopping: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with single tqdm bar showing losses in real-time.

        Args:
            u_train, y_train: Training data
            u_val, y_val: Validation data
            use_early_stopping: Whether to use early stopping

        Returns:
            Dictionary with training history
        """
        # Initialize early stopping
        early_stopping = None
        if use_early_stopping:
            early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )

        # Single progress bar
        pbar = tqdm(range(self.config.num_epochs), desc="Training", ncols=100)
        early_stop_epoch = None

        for epoch in pbar:
            # Train
            train_loss = self.train_epoch(u_train, y_train)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(u_val, y_val)
            self.val_losses.append(val_loss)

            # Update best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.config.save_best_only:
                    self.save_checkpoint(epoch, is_best=True)

            # Update postfix with current losses
            pbar.set_postfix({
                'train': f'{train_loss:.4e}',
                'val': f'{val_loss:.4e}',
                'best': f'{self.best_val_loss:.4e}'
            })

            # Early stopping
            if use_early_stopping:
                if early_stopping(val_loss, self.model):
                    early_stop_epoch = epoch
                    early_stopping.load_best_model(self.model)
                    pbar.close()
                    break

        # Save final checkpoint
        self.save_checkpoint(epoch, is_best=False)

        # Print final report
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total epochs:      {len(self.train_losses)}")
        if early_stop_epoch is not None:
            print(f"Early stopped at:  Epoch {early_stop_epoch}")
        print(f"Final train loss:  {self.train_losses[-1]:.6e}")
        print(f"Final val loss:    {self.val_losses[-1]:.6e}")
        print(f"Best val loss:     {self.best_val_loss:.6e}")
        print("=" * 70 + "\n")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

    @torch.no_grad()
    def predict(self, u: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions.

        Args:
            u: Input tensor

        Returns:
            Model predictions
        """
        self.model.eval()
        u = u.to(self.device)
        y_pred, _ = self.model(u)
        return y_pred.squeeze()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        filepath = self.config.save_dir / filename
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']


# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Visualization utilities for system identification."""

    def __init__(self, save_dir: Path = Path("./figures"), show_plots: bool = True):
        """
        Args:
            save_dir: Directory to save figures
            show_plots: Whether to display plots interactively (default: True)
        """
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.show_plots = show_plots

        # Set publication-quality parameters
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 8,
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.0,
            'figure.dpi': 300,
            'savefig.dpi': 300,
        })

    def plot_losses(
            self,
            train_losses: list,
            val_losses: Optional[list] = None,
            filename: str = 'training_loss.pdf'
    ):
        """
        Plot training and validation losses.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses (optional)
            filename: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(5.25, 1.75))

        epochs = range(len(train_losses))
        ax.plot(epochs, train_losses, label='Training Loss', color='blue', linestyle='-')

        if val_losses is not None:
            ax.plot(epochs, val_losses, label='Validation Loss', color='red', linestyle='--')

        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        filepath = self.save_dir / filename
        plt.savefig(filepath, format='pdf', bbox_inches='tight')

        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_predictions(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            title: str = "Model Predictions",
            filename: str = 'predictions.pdf',
            ylabel: str = r'$y$'
    ):
        """
        Plot true vs predicted outputs.

        Args:
            y_true: True output values
            y_pred: Predicted output values
            title: Plot title
            filename: Filename to save plot
            ylabel: Y-axis label
        """
        fig, ax = plt.subplots(figsize=(6.25, 1.75))

        # Convert to numpy if necessary
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().detach().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().detach().numpy()

        # Plot
        time = np.arange(len(y_true))
        ax.plot(time, y_true, label='True', color='orange', linestyle='-')
        ax.plot(time, y_pred, label='Predicted', color='blue', linestyle=':')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title(title)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        filepath = self.save_dir / filename
        plt.savefig(filepath, format='pdf', bbox_inches='tight')

        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_all_results(
            self,
            train_losses: list,
            val_losses: list,
            y_train: torch.Tensor,
            y_train_pred: torch.Tensor,
            y_val: torch.Tensor,
            y_val_pred: torch.Tensor,
            ylabel: str = r'$y$'
    ):
        """
        Create all plots in one call.

        Args:
            train_losses: Training loss history
            val_losses: Validation loss history
            y_train, y_train_pred: Training data and predictions
            y_val, y_val_pred: Validation data and predictions
            ylabel: Y-axis label for predictions
        """
        # Plot training loss
        self.plot_losses(train_losses, val_losses, 'training_loss.pdf')

        # Plot training predictions
        self.plot_predictions(
            y_train.squeeze(),
            y_train_pred,
            "Training Set Predictions",
            'train_predictions.pdf',
            ylabel
        )

        # Plot validation predictions
        self.plot_predictions(
            y_val.squeeze(),
            y_val_pred,
            "Validation Set Predictions",
            'val_predictions.pdf',
            ylabel
        )


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""

    # Set random seeds for reproducibility
    seed = 9
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Load data
    print("Loading data...")

    train_val, test = nonlinear_benchmarks.Cascaded_Tanks()
    print(test.state_initialization_window_length)  # = 50
    u_train, y_train = train_val
    u_val, y_val = test
    # Convert to torch tensors and reshape to (N, 1)
    u_train = torch.tensor(u_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    u_val = torch.tensor(u_val, dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

    # Initialize configurations
    model_config = ModelConfig(n_u=u_train.shape[1], n_y=y_train.shape[1], param='l2ru', d_model=12, d_state=11,
                               gamma=None, ff='GLU', init='eye',
                               n_layers=3, d_amp=3)
    train_config = TrainingConfig(num_epochs=5000, learning_rate=1e-4)

    # Build model
    print("Building model...")
    ssm_config = model_config.to_ssm_config()
    model = DeepSSM(d_input=model_config.n_u, d_output=model_config.n_y, config=ssm_config)

    # Try RNN
    #model = SimpleLSTM(hidden_dim=32, bidirectional=False, num_layers=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    # Initialize trainer
    trainer = SystemIDTrainer(model, train_config, criterion=nn.MSELoss())

    # Train model
    history = trainer.fit(u_train, y_train, u_val, y_val, use_early_stopping=False)

    # load best model if available
    best_path = train_config.save_dir / "best_model.pth"
    if best_path.exists():
        print(f"Loading best checkpoint from {best_path}")
        trainer.load_checkpoint(str(best_path))
    else:
        print("No best_model.pth found — using final model")

    # Generate predictions
    print("Generating predictions and print best RMSE...")
    y_train_pred = trainer.predict(u_train)
    y_val_pred = trainer.predict(u_val)
    n = test.state_initialization_window_length

    RMSE_result = RMSE(test.y[n:], y_val_pred[n:].cpu().detach().numpy())  # skip the first n
    print(RMSE_result)  # report this number

    # Visualize results
    print("Creating visualizations...")
    visualizer = Visualizer(
        save_dir=train_config.save_dir / "figures",
        show_plots=True
    )

    # Use the convenient plot_all_results method
    visualizer.plot_all_results(
        history['train_losses'],
        history['val_losses'],
        y_train,
        y_train_pred,
        y_val,
        y_val_pred,
        ylabel=r'$h_1$ [cm]'
    )

    # Save loss history
    loss_file = train_config.save_dir / "loss_history.npy"
    np.save(loss_file, np.array(history['train_losses']))
    print(f"Saved loss history to {loss_file}\n")

    x = DeepSSM(d_input=4, d_output=3, d_model=3)


if __name__ == "__main__":
    main()
