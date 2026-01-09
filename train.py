import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Minimal CLI: parse & type-check args for INR training")
    p.add_argument("--arch", required=True, help="Architecture name (must be a simple name, no path)")
    p.add_argument("--data", required=True, help="Path to a .npy data file")
    p.add_argument(
        "--downsample",
        nargs="?",
        const=True,
        type=str2bool,
        default=False,
        help="Whether to downsample (bool). Use as flag or pass true/false",
    )
    p.add_argument("--n_epochs", type=int, default=5000, help="Number of training epochs (int)")
    p.add_argument(
        "--debug",
        nargs="?",
        const=True,
        type=str2bool,
        default=False,
        help="Debug mode (bool). Use as flag or pass true/false",
    )
    return p.parse_args(argv)


def validate_args(args):
    # arch: simple name, no path separators or parent refs
    if not isinstance(args.arch, str) or not args.arch.strip():
        raise ValueError("--arch must be a non-empty string")
    if any(sep in args.arch for sep in ("/", "\\")) or ".." in args.arch:
        raise ValueError("--arch must be a simple name, not a path")

    # data: must exist and be a .npy file
    data_path = Path(args.data)
    if not data_path.exists():
        raise ValueError(f"--data path does not exist: {data_path}")
    if not data_path.is_file():
        raise ValueError(f"--data must be a file: {data_path}")
    if data_path.suffix.lower() != ".npy":
        raise ValueError(f"--data must be a .npy file: {data_path}")

    # n_epochs: positive int
    if not isinstance(args.n_epochs, int) or args.n_epochs <= 0:
        raise ValueError("--n_epochs must be a positive integer")

    # downsample and debug already coerced to bool by type
    if not isinstance(args.downsample, bool):
        raise ValueError("--downsample must be a boolean")
    if not isinstance(args.debug, bool):
        raise ValueError("--debug must be a boolean")

    # optional: warn if arch not present in archs folder
    archs_dir = Path(__file__).resolve().parent / "archs"
    if archs_dir.exists():
        arch_as_dir = archs_dir / args.arch
        arch_as_file = archs_dir / (args.arch + ".py")
        if not (arch_as_dir.exists() or arch_as_file.exists()):
            print(f"Warning: architecture '{args.arch}' not found in archs/", file=sys.stderr)


def prepare_data(
    data_path: str,
    downsample_c: int = 2,
    downsample_t: int = 2,
    seed: int = 123,
):
    data = np.load(data_path).astype(np.float32)
    data = data / np.max(np.abs(data))
    rng = np.random.RandomState(seed)

    # Downsample independently along C and T
    ground_truth = data.copy()
    training_data = data[::downsample_c, ::downsample_t]

    C_cs, N_cs = training_data.shape
    channels_cs = np.linspace(-1, 1, C_cs)
    time_steps = np.linspace(-1, 1, N_cs)
    c_grid_cs, t_grid_cs = np.meshgrid(channels_cs, time_steps, indexing="ij")
    coords_cs = np.stack([c_grid_cs, t_grid_cs], axis=-1).reshape(-1, 2).astype(np.float32)

    full_C, full_N = data.shape
    return ground_truth, training_data, coords_cs, full_C, full_N


def train(args, save_dir="results"):

    # Save directories and logging setup
    model_type = args.arch
    num_epochs = args.n_epochs
    data_path = args.data
    downsample_c = 2 if args.downsample else 1
    downsample_t = 2 if args.downsample else 1
    save_dir = Path(save_dir)
    log_interval = 100 if args.debug else num_epochs + 1
    model_type = args.arch.upper()

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_dir = save_dir / f"run_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ts_dir.mkdir(parents=True, exist_ok=True)
    log_path = ts_dir / "training_log.csv"

    # --- Load and prepare data ---
    ground_truth, training_data, training_coords, full_C, full_N = prepare_data(
        data_path=data_path,
        downsample_c=downsample_c,
        downsample_t=downsample_t,
    )

    plt.imsave(os.path.join(ts_dir, "original_data.png"), ground_truth, cmap="bwr", vmin=-1, vmax=1)
    plt.imsave(os.path.join(ts_dir, "compressed_measurements.png"), training_data, cmap="bwr", vmin=-1, vmax=1)

    # --- Initialize model ---
    if model_type == "SIREN":
        from archs.SIREN import SIREN

        model = SIREN(in_dim=2, hidden_dim=128, hidden_layers=4, out_dim=1).to(device)
    elif model_type == "WIRE":
        from archs.WIRE import WIRE

        model = WIRE(in_dim=2, hidden_dim=128, hidden_layers=4, out_dim=1).to(device)
    elif model_type == "GAUSS":
        from archs.GAUSS import GAUSS

        model = GAUSS(in_dim=2, hidden_dim=128, hidden_layers=4, out_dim=1).to(device)
    elif model_type == "NEWER":
        from archs.NEWER import NEWER

        model = NEWER(in_dim=2, hidden_dim=22, hidden_layers=4).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Training {model_type} | down_c={downsample_c}, down_t={downsample_t}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    decay_epochs = num_epochs // 2
    min_lr = learning_rate * 0.1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(min_lr / learning_rate) ** (1 / decay_epochs))

    # --- Prepare tensors ---
    coords_tensor = torch.from_numpy(training_coords).to(device)
    data_tensor = torch.from_numpy(training_data.reshape(-1, 1)).to(device)

    full_channels = np.linspace(-1, 1, full_C)
    time_steps = np.linspace(-1, 1, full_N)
    c_grid_full, t_grid_full = np.meshgrid(full_channels, time_steps, indexing="ij")
    coords_full = np.stack([c_grid_full, t_grid_full], axis=-1).reshape(-1, 2).astype(np.float32)
    coords_full_tensor = torch.from_numpy(coords_full).to(device)
    # ground_truth_tensor = torch.from_numpy(ground_truth.reshape(-1, 1)).to(device)

    # --- Training loop ---
    loss_list = []
    with open(log_path, "w") as f:
        f.write("epoch,train_loss_mse,full_res_mse\n")

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        outputs = model(coords_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()

        if epoch < decay_epochs:
            scheduler.step()

        loss_list.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4e}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # --- Save logs ---
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                fit_full = model(coords_full_tensor).cpu().numpy().reshape(full_C, full_N)
            full_res_mse = np.mean((fit_full - ground_truth) ** 2)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{loss.item():.6e},{full_res_mse:.6e}\n")
            np.save(os.path.join(ts_dir, f"fit_epoch_{epoch:05d}.npy"), fit_full)
            plt.imsave(os.path.join(ts_dir, f"fit_epoch_{epoch:05d}.png"), fit_full, cmap="bwr")

    return model, loss_list


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_args(args)
    except Exception as e:
        print(f"Argument error: {e}", file=sys.stderr)
        sys.exit(2)

    # Proceed with training
    train(args)


if __name__ == "__main__":
    main()
