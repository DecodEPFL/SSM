import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.io as sio
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, replace
from tqdm import tqdm
import logging
import math
import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
import json
#from SSM.utility import SimpleLSTM
#from neural_ssm import DeepSSM, SSMConfig
from src.neural_ssm.ssm.lru import DeepSSM, SSMConfig, SimpleRNN

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
    num_epochs: int = 8590
    seed: int = 9
    device: str = "cpu" if torch.cuda.is_available() else "cpu"
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
    random_phase = True

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
        )


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
) -> Dict[str, Any]:
    """
    Tune n_layers, d_model, d_state, learning_rate, and gamma with Optuna.

    Returns:
        Dictionary with best hyperparameters and objective value.
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
        trainer = SystemIDTrainer(model, train_config, criterion=nn.MSELoss())
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

    train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
    print(test.state_initialization_window_length)  # = 50
    u_train, y_train = train_val
    u_val, y_val = test


    # Convert to torch tensors and reshape to (N, 1)
    u_train = torch.tensor(u_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    u_val = torch.tensor(u_val, dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
    #
    # u_train = _normalize_to_3d(u_train)
    # u_val = _normalize_to_3d(u_val)
    # y_val = _normalize_to_3d(y_val)
    # y_train = _normalize_to_3d(y_train)

    # # data normalization
    # mu_u = u_train.mean(dim=0)  # over all time + trajectories
    # mu_y = y_train.mean(dim=0)
    # u_center = u_train - mu_u
    # y_center = y_train - mu_y
    # sigma_u = u_train.sub(mu_u).pow(2).mean(dim=0).sqrt()
    # sigma_y = y_train.sub(mu_y).pow(2).mean(dim=0).sqrt()
    # u_norm = u_center / sigma_u
    # y_norm = y_center / sigma_y
    #
    # u_val_norm = (u_val - mu_u) / (sigma_u + 1e-8)
    # y_val_norm = (y_val - mu_y) / (sigma_y + 1e-8)


    # Initialize configurations
    model_config = ModelConfig(n_u=u_train.shape[1], n_y=y_train.shape[1], param='tv', d_model=8, d_state=8,
                               gamma=None, ff='GLU', init='eye',
                               n_layers=6, d_amp=3, rho=0.9, phase_center=0.0, max_phase_b=0.04, d_hidden=12, nl_layers=3)
    train_config = TrainingConfig(num_epochs=6000, learning_rate=1.6568e-02)
    hpo_config = HyperOptConfig(enabled=False, n_trials=20, num_epochs=250)

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
    model = DeepSSM(d_input=model_config.n_u, d_output=model_config.n_y, config=ssm_config)

    #model = SimpleRNN(d_input=1, d_hidden=20, num_layers=1, d_output= 1, nonlinearity='tanh')


    # Try RNN
    #model = SimpleLSTM(hidden_dim=32, bidirectional=False, num_layers=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    # Initialize trainer
    trainer = SystemIDTrainer(model, train_config, criterion=nn.MSELoss())

    # Train model
    history = trainer.fit(u_train, y_train, u_val, y_val, use_early_stopping=False)

    # Train model (normalized)
    #history = trainer.fit(u_norm, y_norm, u_val_norm, y_val_norm, use_early_stopping=False)

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

    RMSE_result = 1000*RMSE(test.y[n:], y_val_pred[n:].cpu().detach().numpy())  # skip the first n
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
