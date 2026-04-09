import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.io as sio
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, replace, field
import copy
from tqdm import tqdm
import logging
import math
import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
import json
from src.neural_ssm.ssm import DeepSSM, SSMConfig, SimpleRNN
from src.neural_ssm.rens.ren import REN

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None



# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 2000
    seed: int = 9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Validation init window (timesteps skipped in loss computation)
    init_window: int = 50

    # Early stopping
    patience: int = 50
    min_delta: float = 1e-6

    # Checkpointing
    save_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "checkpoints")
    save_best_only: bool = True

    # Data normalization
    normalize_data: bool = False
    normalization_eps: float = 1e-6


class RENWrapper(nn.Module):
    """Adapts REN to the (B, T, n_u) → (y, None) interface expected by SystemIDTrainer.

    REN.forward() processes one timestep at a time using internal state self.x.
    This wrapper resets the state, rebuilds the constrained matrices from the
    current learnable parameters, then loops over the time axis.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_internal: int, dim_nl: int,
                 **kwargs):
        super().__init__()
        self.ren = REN(dim_in=dim_in, dim_out=dim_out,
                       dim_internal=dim_internal, dim_nl=dim_nl, **kwargs)

    def forward(self, u: torch.Tensor):
        # Accept (T, n_u) or (B, T, n_u); always run as (B, T, n_u).
        if u.dim() == 2:
            u = u.unsqueeze(0)  # (T, n_u) → (1, T, n_u)
        B = u.shape[0]
        self.ren.x = torch.zeros(B, 1, self.ren.dim_internal, device=u.device, dtype=u.dtype)
        y = self.ren(u)  # (B, T, n_y)
        if B == 1:
            y = y.squeeze(0)  # back to (T, n_y) to match DeepSSM's output convention
        return y, None


class LSTMWrapper(nn.Module):
    """Thin PyTorch LSTM baseline with the same interface as DeepSSM."""

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            dim_hidden: int = 64,
            num_layers: int = 2,
            dropout: float = 0.0,
            bias: bool = True,
            bidirectional: bool = False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=dim_in,
            hidden_size=dim_hidden,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(dim_hidden * self.num_directions, dim_out, bias=bias)

    def forward(self, u: torch.Tensor):
        # Accept (T, n_u) or (B, T, n_u); always run as (B, T, n_u).
        squeeze_batch = False
        if u.dim() == 2:
            u = u.unsqueeze(0)
            squeeze_batch = True

        y_seq, _ = self.lstm(u)
        y = self.out_proj(y_seq)

        if squeeze_batch:
            y = y.squeeze(0)

        return y, None


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    n_u: int = 1
    n_y: int = 1
    d_model: int = 16
    d_state: int = 11
    n_layers: int = 1
    ff: str = "LMLP"  # GLU | MLP | LMLP
    max_phase: float = math.pi / 60
    r_min: float = 0.7
    r_max: float = 0.98
    d_amp: int = 8
    param: str = 'l2ru'
    d_hidden: int = 8
    nl_layers: int = 3
    gamma: Optional[float] = 2
    init: str = 'rand'
    rho: float = 0.9
    max_phase_b: float = 0.5  # small spread
    phase_center: float = 0  # center angle ≈ 17°
    random_phase: bool = True
    learn_x0: bool = True

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
            d_hidden = self.d_hidden,
            nl_layers = self.nl_layers,
            dim_amp=self.d_amp,
            param=self.param,
            gamma=self.gamma,
            init=self.init,
            rho = self.rho,
            max_phase_b = self.max_phase_b,
            phase_center = self.phase_center,
            random_phase=self.random_phase,
            learn_x0= self.learn_x0
        )


@dataclass
class LivePlotConfig:
    """Configuration for real-time training animation (scalar outputs only)."""
    enabled: bool = False
    update_interval: int = 1    # redraw every N epochs
    n_points: int = 1200        # max time steps shown (subsample for speed)
    figsize: tuple = (13, 6)
    show_validation: bool = True
    transition_frames: int = 4
    transition_pause: float = 0.0015


@dataclass
class HyperOptConfig:
    """Configuration for Optuna hyperparameter optimization."""
    enabled: bool = True
    n_trials: int = 20
    num_epochs: int = 250
    timeout_seconds: Optional[int] = None
    n_layers_min: int = 1
    n_layers_max: int = 6
    lr_min: float = 1e-4
    lr_max: float = 5e-2
    gamma_min: float = 0.5
    gamma_max: float = 20.0
    d_model_min: int = 4
    d_model_max: int = 32
    d_state_min: int = 4
    d_state_max: int = 32
    d_state_step: int = 2


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


@dataclass
class ChannelwiseStandardizer:
    """Per-channel affine normalization using training-split statistics only."""
    u_mean: torch.Tensor
    u_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    @staticmethod
    def _reduce_dims(x: torch.Tensor) -> Tuple[int, ...]:
        if x.dim() < 2:
            raise ValueError(
                "Expected input with at least one sample axis and one feature axis; "
                f"got shape {tuple(x.shape)}"
            )
        return tuple(range(x.dim() - 1))

    @classmethod
    def fit(
            cls,
            u_train: torch.Tensor,
            y_train: torch.Tensor,
            eps: float = 1e-6,
    ) -> "ChannelwiseStandardizer":
        reduce_u = cls._reduce_dims(u_train)
        reduce_y = cls._reduce_dims(y_train)
        u_mean = u_train.mean(dim=reduce_u, keepdim=True)
        u_std = u_train.std(dim=reduce_u, keepdim=True, unbiased=False).clamp_min(eps)
        y_mean = y_train.mean(dim=reduce_y, keepdim=True)
        y_std = y_train.std(dim=reduce_y, keepdim=True, unbiased=False).clamp_min(eps)
        return cls(
            u_mean=u_mean.detach().cpu(),
            u_std=u_std.detach().cpu(),
            y_mean=y_mean.detach().cpu(),
            y_std=y_std.detach().cpu(),
        )

    def _match(self, tensor: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return stats.to(device=tensor.device, dtype=tensor.dtype)

    def transform_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self._match(u, self.u_mean)) / self._match(u, self.u_std)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self._match(y, self.y_mean)) / self._match(y, self.y_std)

    def inverse_transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._match(y, self.y_std) + self._match(y, self.y_mean)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "u_mean": self.u_mean,
            "u_std": self.u_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "ChannelwiseStandardizer":
        return cls(
            u_mean=state["u_mean"].detach().cpu(),
            u_std=state["u_std"].detach().cpu(),
            y_mean=state["y_mean"].detach().cpu(),
            y_std=state["y_std"].detach().cpu(),
        )


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
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
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
            optimizer: Optional[torch.optim.Optimizer] = None,
            live_plot_config: Optional[LivePlotConfig] = None,
            normalizer: Optional[ChannelwiseStandardizer] = None,
    ):
        """
        Args:
            model: The model to train
            train_config: Training configuration
            criterion: Loss function (defaults to MSELoss)
            optimizer: Optimizer (defaults to Adam)
            live_plot_config: If provided and enabled, shows a real-time
                animated plot of predictions vs target (scalar outputs only).
        """
        self.model = model
        self.config = train_config
        self.device = torch.device(train_config.device)
        self.live_plot_config = live_plot_config
        self.normalizer = normalizer
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

    def _normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        return self.normalizer.transform_u(u) if self.normalizer is not None else u

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.normalizer.transform_y(y) if self.normalizer is not None else y

    def _denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.normalizer.inverse_transform_y(y) if self.normalizer is not None else y

    def train_epoch(self, u: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Train for one epoch.

        Args:
            u: Input tensor
            y: Target tensor

        Returns:
            Average training loss and model predictions for the current epoch
        """
        self.model.train()

        # Move data to device
        u = u.to(self.device)
        y = y.to(self.device)
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        # Forward pass
        y_pred_norm, _ = self.model(u_norm)
        y_pred = self._denormalize_y(y_pred_norm).squeeze()
        y_norm = y_norm.squeeze()

        # Compute loss
        loss = self.criterion(y_pred_norm.squeeze(), y_norm)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), y_pred.detach().cpu()

    @torch.no_grad()
    def validate(self, u: torch.Tensor, y: torch.Tensor, n: int = 50) -> Tuple[float, torch.Tensor]:
        """
        Validate the model.

        Args:
            n: Initialization window length
            u: Input tensor
            y: Target tensor

        Returns:
            Validation loss and validation predictions
        """
        self.model.eval()

        # Move data to device
        u = u.to(self.device)
        y = y.to(self.device)
        u_norm = self._normalize_u(u)
        y_norm = self._normalize_y(y)

        # Forward pass
        y_pred_norm, _ = self.model(u_norm)
        y_pred = self._denormalize_y(y_pred_norm).squeeze()
        y_norm = y_norm.squeeze()

        # Compute loss
        loss = self.criterion(y_pred_norm.squeeze()[n:], y_norm[n:])

        return loss.item(), y_pred.detach().cpu()

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

        # Live plot (scalar outputs only — disable via LivePlotConfig.enabled=False)
        live_plotter = None
        cfg = self.live_plot_config
        if cfg is not None and cfg.enabled:
            live_plotter = LivePlotter(
                config=cfg,
                y_train_true=y_train,
                y_val_true=y_val,
                total_epochs=self.config.num_epochs,
                init_window=self.config.init_window,
            )

        # Progress bar
        pbar = tqdm(range(self.config.num_epochs), desc="Training", ncols=100)
        early_stop_epoch = None

        for epoch in pbar:
            # Train
            train_loss, train_pred = self.train_epoch(u_train, y_train)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_pred = self.validate(u_val, y_val, n=self.config.init_window)
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

            # Live plot update
            if live_plotter is not None and epoch % cfg.update_interval == 0:
                live_plotter.update(
                    epoch=epoch,
                    train_pred=train_pred,
                    val_pred=val_pred,
                    train_losses=self.train_losses,
                    val_losses=self.val_losses,
                    best_val_loss=self.best_val_loss,
                )

            # Early stopping
            if use_early_stopping:
                if early_stopping(val_loss, self.model):
                    early_stop_epoch = epoch
                    early_stopping.load_best_model(self.model)
                    pbar.close()
                    break

        # Final live plot update and close
        if live_plotter is not None:
            final_train_pred = self.predict(u_train).detach().cpu()
            final_val_pred = self.predict(u_val).detach().cpu()
            live_plotter.update(
                epoch=epoch,
                train_pred=final_train_pred,
                val_pred=final_val_pred,
                train_losses=self.train_losses,
                val_losses=self.val_losses,
                best_val_loss=self.best_val_loss,
            )
            live_plotter.close()

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
        u_norm = self._normalize_u(u)
        y_pred_norm, _ = self.model(u_norm)
        y_pred = self._denormalize_y(y_pred_norm)
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
            'best_val_loss': self.best_val_loss,
            'normalizer_state_dict': None if self.normalizer is None else self.normalizer.state_dict(),
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
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        normalizer_state = checkpoint.get('normalizer_state_dict')
        if normalizer_state is not None:
            self.normalizer = ChannelwiseStandardizer.from_state_dict(normalizer_state)


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

    @staticmethod
    def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 1:
            return x[:, None]
        if x.ndim == 2:
            return x
        if x.ndim == 3 and x.shape[0] == 1:
            return x[0]
        raise ValueError(f"Expected a single sequence shaped (T,), (T, C), or (1, T, C); got {x.shape}")

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

    def plot_inputs(
            self,
            u_train: torch.Tensor,
            u_val: torch.Tensor,
            title: str = "Input Signals",
            filename: str = 'input_signals',
            ylabel: str = r'$u$',
            init_window: int = 0,
    ):
        """
        Plot training and validation inputs side by side.

        Args:
            u_train: Training input sequence
            u_val: Validation input sequence
            title: Plot title
            filename: Base filename used for saved figures
            ylabel: Y-axis label
            init_window: Optional validation initialization window to highlight
        """
        train = self._to_numpy_2d(u_train)
        val = self._to_numpy_2d(u_val)
        n_channels = max(train.shape[1], val.shape[1])

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6.25, 3.2),
            sharex=False,
            constrained_layout=True,
        )
        fig.suptitle(title)

        palette = ['#33658a', '#2f9e44', '#ef476f', '#8d6a9f', '#f4a261', '#2a9d8f']
        for ax, data, split_name in zip(axes, (train, val), ("Training input", "Validation input")):
            time = np.arange(data.shape[0])
            for ch in range(data.shape[1]):
                label = f'$u_{ch + 1}$' if n_channels > 1 else 'Input'
                ax.plot(time, data[:, ch], color=palette[ch % len(palette)], label=label)
            if split_name == "Validation input" and init_window > 0:
                ax.axvspan(0, init_window, color='#cbd5e1', alpha=0.35, lw=0, label='Init window')
            ax.set_title(split_name)
            ax.set_xlabel('Time step')
            ax.set_ylabel(ylabel)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right', frameon=False, ncol=min(max(data.shape[1], 1), 3))

        pdf_path = self.save_dir / f'{filename}.pdf'
        png_path = self.save_dir / f'{filename}.png'
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(png_path, format='png', bbox_inches='tight')

        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_all_results(
            self,
            train_losses: list,
            val_losses: list,
            u_train: torch.Tensor,
            u_val: torch.Tensor,
            y_train: torch.Tensor,
            y_train_pred: torch.Tensor,
            y_val: torch.Tensor,
            y_val_pred: torch.Tensor,
            ylabel: str = r'$y$',
            ulabel: str = r'$u$',
            init_window: int = 0,
    ):
        """
        Create all plots in one call.

        Args:
            train_losses: Training loss history
            val_losses: Validation loss history
            u_train, u_val: Training and validation inputs
            y_train, y_train_pred: Training data and predictions
            y_val, y_val_pred: Validation data and predictions
            ylabel: Y-axis label for predictions
        """
        # Plot training loss
        self.plot_losses(train_losses, val_losses, 'training_loss.pdf')

        # Plot inputs
        self.plot_inputs(
            u_train=u_train,
            u_val=u_val,
            title="Training and Validation Inputs",
            filename='input_signals',
            ylabel=ulabel,
            init_window=init_window,
        )

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
# Live Training Plotter
# ============================================================================

class LivePlotter:
    """
    Real-time animated plot of train/validation predictions and losses.
    Only meaningful for scalar (n_y=1) outputs; disable for multi-output datasets.

    Uses blitting for smooth updates: static axes are drawn once and cached;
    only the animated artists are redrawn each frame. Prediction updates are
    tweened across a few subframes so the motion feels less abrupt.
    """

    def __init__(
        self,
        config: LivePlotConfig,
        y_train_true: torch.Tensor,
        y_val_true: Optional[torch.Tensor],
        total_epochs: int,
        init_window: int = 0,
    ):
        self.config = config
        self.y_train_true = y_train_true.squeeze().detach().cpu().numpy()
        self.y_val_true = (
            y_val_true.squeeze().detach().cpu().numpy()
            if y_val_true is not None else None
        )
        self._total_epochs = total_epochs
        self._init_window = init_window
        self._show_validation = config.show_validation and self.y_val_true is not None

        self._idx_train, self._y_train_plot, self._train_ylim = self._prepare_series(self.y_train_true)
        if self._show_validation:
            self._idx_val, self._y_val_plot, self._val_ylim = self._prepare_series(self.y_val_true)
        else:
            self._idx_val, self._y_val_plot, self._val_ylim = None, None, None

        # Loss y-limits: track the running range, expand-only (never shrink)
        self._loss_ylim = (np.inf, -np.inf)
        self._prev_train_pred = None
        self._prev_val_pred = None

        try:
            # Force a real GUI window — PyCharm's inline backend captures figures
            # statically and cannot update live. switch_backend() is safe to call
            # after pyplot is already imported. MacOSX is the native macOS backend;
            # TkAgg/Qt5Agg are cross-platform fallbacks.
            for _backend in ("MacOSX", "TkAgg", "Qt5Agg", "QtAgg"):
                try:
                    plt.switch_backend(_backend)
                    break
                except Exception:
                    continue

            plt.ion()
            self._fig = plt.figure(figsize=config.figsize)
            self._fig.patch.set_facecolor("#0b1118")
            self._fig.suptitle("Live Training Progress", fontsize=11, color="#f6f7fb")

            gs = self._fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.28, wspace=0.18)
            if self._show_validation:
                self._ax_train = self._fig.add_subplot(gs[0, 0])
                self._ax_val = self._fig.add_subplot(gs[0, 1])
            else:
                self._ax_train = self._fig.add_subplot(gs[0, :])
                self._ax_val = None
            self._ax_loss = self._fig.add_subplot(gs[1, :])

            self._line_train_glow, self._line_train = self._configure_prediction_axis(
                self._ax_train,
                title="Training Sequence",
                idx=self._idx_train,
                y_true=self._y_train_plot,
                ylim=self._train_ylim,
                pred_color="#5ec8e5",
            )
            if self._ax_val is not None:
                self._line_val_glow, self._line_val = self._configure_prediction_axis(
                    self._ax_val,
                    title="Validation Sequence",
                    idx=self._idx_val,
                    y_true=self._y_val_plot,
                    ylim=self._val_ylim,
                    pred_color="#ff7a90",
                    show_init_window=True,
                )
            else:
                self._line_val_glow = None
                self._line_val = None

            self._configure_loss_axis(total_epochs)

            self._epoch_text = self._ax_train.text(
                0.02, 0.96, "", transform=self._ax_train.transAxes,
                fontsize=9, va="top", color="#e8edf6", animated=True,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#101826", edgecolor="none", alpha=0.82),
            )
            self._stats_text = self._ax_loss.text(
                0.015, 0.94, "", transform=self._ax_loss.transAxes,
                fontsize=8, va="top", color="#e8edf6", animated=True,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#101826", edgecolor="none", alpha=0.82),
            )

            self._fig.tight_layout()

            # Full draw once to render the static content (axes, labels, target line)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

            # Cache the static background for blitting
            self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)
            self._fig.canvas.mpl_connect("resize_event", lambda _event: self._redraw_background())
            self._active = True
        except Exception as e:
            print(f"[LivePlotter] Could not initialise interactive plot: {e}")
            self._active = False

    def _prepare_series(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """Subsample a scalar series and compute fixed y-limits for plotting."""
        T = len(series)
        idx = (
            np.linspace(0, T - 1, self.config.n_points, dtype=int)
            if T > self.config.n_points else np.arange(T)
        )
        series_plot = series[idx]
        series_min = float(series_plot.min())
        series_max = float(series_plot.max())
        margin = max((series_max - series_min) * 0.12, 1e-4)
        return idx, series_plot, (series_min - margin, series_max + margin)

    def _style_axis(self, ax: plt.Axes):
        ax.set_facecolor("#111925")
        ax.grid(True, color="#425066", alpha=0.24, linewidth=0.6)
        ax.tick_params(colors="#d7dde8", labelsize=8)
        ax.xaxis.label.set_color("#eef2f7")
        ax.yaxis.label.set_color("#eef2f7")
        ax.title.set_color("#f6f7fb")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#5d6c80")

    def _configure_prediction_axis(
        self,
        ax: plt.Axes,
        title: str,
        idx: np.ndarray,
        y_true: np.ndarray,
        ylim: Tuple[float, float],
        pred_color: str,
        show_init_window: bool = False,
    ):
        self._style_axis(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Output")
        ax.set_xlim(int(idx[0]), int(idx[-1]))
        ax.set_ylim(*ylim)

        ax.plot(idx, y_true, color="#f6bd60", lw=1.1, alpha=0.88, label="Target")
        if show_init_window and self._init_window > 0:
            ax.axvspan(0, self._init_window, color="#d7dde8", alpha=0.08, lw=0)
            ax.text(
                0.02, 0.08, f"warm-up ignored in loss: {self._init_window}",
                transform=ax.transAxes, fontsize=7, color="#b5c0cf",
            )
        ax.legend(fontsize=7, loc="upper right", facecolor="#111925", edgecolor="#334155", labelcolor="#eef2f7")

        glow_line = ax.plot(
            idx, np.full_like(y_true, np.nan),
            color=pred_color, lw=4.5, alpha=0.14, animated=True,
            solid_capstyle="round",
        )[0]
        pred_line = ax.plot(
            idx, np.full_like(y_true, np.nan),
            color=pred_color, lw=1.8, alpha=0.97, animated=True,
            solid_capstyle="round",
        )[0]
        return glow_line, pred_line

    def _configure_loss_axis(self, total_epochs: int):
        self._style_axis(self._ax_loss)
        self._ax_loss.set_title("Loss Trajectory", fontsize=10)
        self._ax_loss.set_ylabel("Loss")
        self._ax_loss.set_xlabel("Epoch")
        self._ax_loss.set_yscale("log")
        self._ax_loss.set_xlim(0, max(1, total_epochs - 1))

        (self._line_tloss_glow,) = self._ax_loss.plot(
            [], [], color="#5ec8e5", lw=4.0, alpha=0.12, animated=True,
        )
        (self._line_tloss,) = self._ax_loss.plot(
            [], [], color="#5ec8e5", lw=1.3, animated=True, label="Train",
        )
        (self._line_vloss_glow,) = self._ax_loss.plot(
            [], [], color="#ff7a90", lw=4.0, alpha=0.12, animated=True,
        )
        (self._line_vloss,) = self._ax_loss.plot(
            [], [], color="#ff7a90", lw=1.3, linestyle="--", animated=True, label="Val",
        )
        (self._best_marker,) = self._ax_loss.plot(
            [], [], linestyle="None", marker="o", markersize=5.5,
            color="#9bffb0", markeredgecolor="#ffffff", markeredgewidth=0.5,
            animated=True, label="Best val",
        )
        self._ax_loss.legend(fontsize=7, loc="upper right", facecolor="#111925", edgecolor="#334155", labelcolor="#eef2f7")

    def _redraw_background(self):
        """Full redraw + re-cache background (called only when axes limits change)."""
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)

    def _to_plot_values(self, pred: torch.Tensor, idx: np.ndarray) -> np.ndarray:
        pred_np = pred.squeeze().detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred).squeeze()
        return pred_np[idx]

    def _blend(self, previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
        if previous is None or previous.shape != current.shape:
            return current
        return previous + alpha * (current - previous)

    def _draw_artists(
        self,
        train_plot: np.ndarray,
        val_plot: Optional[np.ndarray],
        epoch: int,
        train_losses: list,
        val_losses: list,
        best_val_loss: float,
    ):
        self._fig.canvas.restore_region(self._bg)

        self._line_train_glow.set_ydata(train_plot)
        self._line_train.set_ydata(train_plot)
        self._ax_train.draw_artist(self._line_train_glow)
        self._ax_train.draw_artist(self._line_train)

        if self._ax_val is not None and val_plot is not None:
            self._line_val_glow.set_ydata(val_plot)
            self._line_val.set_ydata(val_plot)
            self._ax_val.draw_artist(self._line_val_glow)
            self._ax_val.draw_artist(self._line_val)

        epochs_x = np.arange(len(train_losses))
        self._line_tloss_glow.set_data(epochs_x, train_losses)
        self._line_tloss.set_data(epochs_x, train_losses)
        self._line_vloss_glow.set_data(epochs_x, val_losses)
        self._line_vloss.set_data(epochs_x, val_losses)

        best_epoch = int(np.argmin(val_losses)) if val_losses else 0
        best_value = float(val_losses[best_epoch]) if val_losses else best_val_loss
        self._best_marker.set_data([best_epoch], [best_value])

        self._epoch_text.set_text(f"Epoch {epoch + 1}/{self._total_epochs}")
        self._stats_text.set_text(
            f"train {train_losses[-1]:.2e}   val {val_losses[-1]:.2e}\n"
            f"best {best_val_loss:.2e} @ epoch {best_epoch + 1}"
        )

        self._ax_loss.draw_artist(self._line_tloss_glow)
        self._ax_loss.draw_artist(self._line_tloss)
        self._ax_loss.draw_artist(self._line_vloss_glow)
        self._ax_loss.draw_artist(self._line_vloss)
        self._ax_loss.draw_artist(self._best_marker)
        self._ax_train.draw_artist(self._epoch_text)
        self._ax_loss.draw_artist(self._stats_text)

        self._fig.canvas.blit(self._fig.bbox)
        self._fig.canvas.flush_events()

    def _pause_between_frames(self):
        if self.config.transition_pause <= 0:
            return
        try:
            self._fig.canvas.start_event_loop(self.config.transition_pause)
        except Exception:
            plt.pause(self.config.transition_pause)

    @torch.no_grad()
    def update(
        self,
        epoch: int,
        train_pred: torch.Tensor,
        val_pred: Optional[torch.Tensor],
        train_losses: list,
        val_losses: list,
        best_val_loss: float,
    ):
        if not self._active:
            return

        train_plot = self._to_plot_values(train_pred, self._idx_train)
        val_plot = (
            self._to_plot_values(val_pred, self._idx_val)
            if self._show_validation and val_pred is not None else None
        )

        # Check if loss ylim needs expanding (expand-only, never shrink)
        finite = [v for v in train_losses + val_losses if np.isfinite(v) and v > 0]
        needs_bg_refresh = False
        if finite:
            new_ymin = min(finite) * 0.5
            new_ymax = max(finite) * 2.0
            lo, hi = self._loss_ylim
            if new_ymin < lo or new_ymax > hi:
                lo = min(lo, new_ymin)
                hi = max(hi, new_ymax)
                self._loss_ylim = (lo, hi)
                self._ax_loss.set_ylim(lo, hi)
                needs_bg_refresh = True

        if needs_bg_refresh:
            self._redraw_background()

        n_frames = max(1, int(self.config.transition_frames))
        for frame_idx in range(1, n_frames + 1):
            alpha = frame_idx / n_frames
            frame_train = self._blend(self._prev_train_pred, train_plot, alpha)
            frame_val = self._blend(self._prev_val_pred, val_plot, alpha) if val_plot is not None else None
            self._draw_artists(
                train_plot=frame_train,
                val_plot=frame_val,
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                best_val_loss=best_val_loss,
            )
            if frame_idx < n_frames:
                self._pause_between_frames()

        self._prev_train_pred = train_plot.copy()
        self._prev_val_pred = val_plot.copy() if val_plot is not None else None

    def close(self):
        if self._active:
            plt.ioff()
            self._active = False


# ============================================================================
# Main Training Script
# ============================================================================

def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def optimize_hyperparameters(
        u_train: torch.Tensor,
        y_train: torch.Tensor,
        u_val: torch.Tensor,
        y_val: torch.Tensor,
        init_window: int,
        base_model_config: ModelConfig,
        base_train_config: TrainingConfig,
        hpo_config: HyperOptConfig,
        normalizer: Optional[ChannelwiseStandardizer] = None,
) -> Dict[str, Any]:
    """
    Tune n_layers, d_model, d_state, learning_rate, and gamma with Optuna.

    Returns:
        Dictionary with the best hyperparameters and objective value.
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Install it with: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        n_layers = trial.suggest_int("n_layers", hpo_config.n_layers_min, hpo_config.n_layers_max)
        d_model = trial.suggest_int("d_model", hpo_config.d_model_min, hpo_config.d_model_max)
        d_state = trial.suggest_int(
            "d_state",
            hpo_config.d_state_min,
            hpo_config.d_state_max,
            step=hpo_config.d_state_step,
        )
        learning_rate = trial.suggest_float(
            "learning_rate", hpo_config.lr_min, hpo_config.lr_max, log=True
        )
        gamma = trial.suggest_float("gamma", hpo_config.gamma_min, hpo_config.gamma_max, log=True)

        if base_model_config.param == "l2n" and d_state % 2 != 0:
            raise optuna.TrialPruned("d_state must be even when param='l2n'.")

        model_config = replace(
            base_model_config,
            n_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            gamma=gamma,
        )
        trial_save_dir = base_train_config.save_dir / "optuna_trials" / f"trial_{trial.number}"
        train_config = replace(
            base_train_config,
            learning_rate=learning_rate,
            num_epochs=hpo_config.num_epochs,
            save_best_only=False,
            save_dir=trial_save_dir,
        )

        set_random_seed(base_train_config.seed + trial.number)

        model = DeepSSM(
            d_input=model_config.n_u,
            d_output=model_config.n_y,
            config=model_config.to_ssm_config(),
        )
        trainer = SystemIDTrainer(
            model,
            train_config,
            criterion=nn.MSELoss(),
            normalizer=normalizer,
        )
        trainer.fit(u_train, y_train, u_val, y_val, use_early_stopping=True)

        y_val_target = y_val.squeeze().cpu().detach().numpy()
        y_val_pred = trainer.predict(u_val).squeeze().cpu().detach().numpy()
        val_rmse = 1000 * RMSE(y_val_target[init_window:], y_val_pred[init_window:])

        trial.set_user_attr("best_val_loss", trainer.best_val_loss)
        return float(val_rmse)

    sampler = optuna.samplers.TPESampler(seed=base_train_config.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=hpo_config.n_trials,
        timeout=hpo_config.timeout_seconds,
    )

    best_params = {
        "n_layers": int(study.best_params["n_layers"]),
        "d_model": int(study.best_params["d_model"]),
        "d_state": int(study.best_params["d_state"]),
        "learning_rate": float(study.best_params["learning_rate"]),
        "gamma": float(study.best_params["gamma"]),
        "objective_rmse_x1000": float(study.best_value),
    }
    return best_params


def main():
    """Main training function."""

    # Set random seeds for reproducibility
    seed = 9
    set_random_seed(seed)

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

    # ---- Toggle live plot here (disable for multi-output datasets) ----
    live_plot_config = LivePlotConfig(
        enabled=True,          # <-- set to False to turn off
        update_interval=1,     # redraw every epoch for the smoothest motion
        n_points=1200,         # max time steps shown
        transition_frames=5,
        transition_pause=0.002,
    )

    # Initialize configurations
    model_config = ModelConfig(n_u=u_train.shape[1], n_y=y_train.shape[1], param='l2ru', d_model=5, d_state=8,
                               gamma= 40, ff='TLIP', init='eye',
                               n_layers=3, d_amp=3, rho=0.99, phase_center=0.0, max_phase_b=2*np.pi, d_hidden=12, nl_layers=3, learn_x0=True)
    train_config = TrainingConfig(
        num_epochs=2000,
        learning_rate=1.6568e-02,
        init_window=test.state_initialization_window_length,
        normalize_data=False,  # set True to enable train-split z-score normalization
    )
    hpo_config = HyperOptConfig(enabled=False, n_trials=20, num_epochs=250)

    normalizer = None
    if train_config.normalize_data:
        normalizer = ChannelwiseStandardizer.fit(
            u_train,
            y_train,
            eps=train_config.normalization_eps,
        )

        def _fmt_stats(t: torch.Tensor) -> str:
            values = t.squeeze().detach().cpu().numpy()
            return np.array2string(np.atleast_1d(values), precision=4, separator=", ")

        print("Using training-split z-score normalization:")
        print(f"  u_mean = {_fmt_stats(normalizer.u_mean)}")
        print(f"  u_std  = {_fmt_stats(normalizer.u_std)}")
        print(f"  y_mean = {_fmt_stats(normalizer.y_mean)}")
        print(f"  y_std  = {_fmt_stats(normalizer.y_std)}")
    else:
        print("Using raw signals without dataset normalization.")

    # Hyperparameter search
    if hpo_config.enabled:
        print("Running Optuna hyperparameter optimization...")
        best_params = optimize_hyperparameters(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            init_window=test.state_initialization_window_length,
            base_model_config=model_config,
            base_train_config=train_config,
            hpo_config=hpo_config,
            normalizer=normalizer,
        )
        model_config = replace(
            model_config,
            n_layers=best_params["n_layers"],
            d_model=best_params["d_model"],
            d_state=best_params["d_state"],
            gamma=best_params["gamma"],
        )
        train_config = replace(train_config, learning_rate=best_params["learning_rate"])

        print("Best hyperparameters found:")
        print(f"  n_layers:      {best_params['n_layers']}")
        print(f"  d_model:       {best_params['d_model']}")
        print(f"  d_state:       {best_params['d_state']}")
        print(f"  learning_rate: {best_params['learning_rate']:.4e}")
        print(f"  gamma:         {best_params['gamma']:.4f}")
        print(f"  RMSE x1000:    {best_params['objective_rmse_x1000']:.6f}")

        best_params_path = train_config.save_dir / "optuna_best_params.json"
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved Optuna results to {best_params_path}")

        # Reset seed before final training for reproducibility
        set_random_seed(seed)

    # Build model
    print("Building model...")
    ssm_config = model_config.to_ssm_config()

    # --- DeepSSM (comment out to use REN instead) ---
    model = DeepSSM(d_input=model_config.n_u, d_output=model_config.n_y, config=ssm_config)

    # --- LSTM reference baseline (comment out to use DeepSSM instead) ---
    # model = LSTMWrapper(
    #     dim_in=model_config.n_u,
    #     dim_out=model_config.n_y,
    #     dim_hidden=32,
    #     num_layers=2,
    #     dropout=0.1,
    #     bidirectional=False,
    # )

    #--- REN (comment out to use DeepSSM instead) ---
    # model = RENWrapper(
    #     dim_in=model_config.n_u,
    #     dim_out=model_config.n_y,
    #     dim_internal=model_config.d_state,
    #     dim_nl=model_config.d_state,
    # )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    # Initialize trainer
    trainer = SystemIDTrainer(model, train_config, criterion=nn.MSELoss(),
                              live_plot_config=live_plot_config,
                              normalizer=normalizer)

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
        u_train,
        u_val,
        y_train,
        y_train_pred,
        y_val,
        y_val_pred,
        ylabel=r'$h_1$ [cm]',
        ulabel=r'$q$',
        init_window=n,
    )

    # Save loss history
    loss_file = train_config.save_dir / "loss_history.npy"
    np.save(loss_file, np.array(history['train_losses']))
    print(f"Saved loss history to {loss_file}\n")

if __name__ == "__main__":
    main()
