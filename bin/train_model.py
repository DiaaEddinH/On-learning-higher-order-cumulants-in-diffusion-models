import json
import os

import numpy as np
import torch
import torch.distributed as dist
from DiffusionModels import (
    DriftScoreModel,
    MarginalProb,
    Net,
    ScoreModel,
    VPMarginalProb,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm, trange


def set_device(device: str = "cpu") -> torch.device:
    assert device in ["gpu", "cpu"], f"{device} is not a supported device"
    if device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Neither CUDA nor MPS are available. Setting CPU as device.")
    return torch.device("cpu")


def ddp_setup(device):
    """
    Setup for DistributedDataParallel training using torchrun
    """
    if device == "cpu":
        backend = "gloo"
    else:
        backend = "nccl"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=backend)


def train_model(
    data,
    model: ScoreModel,
    marginal_prob: MarginalProb,
    params: dict,
    filepath: str,
    device: torch.device,
    num_workers: int = 0,
):
    assert str(device) in [
        "cpu",
        "cuda",
        "mps",
    ], f"{device} is not supported for training"
    torch.manual_seed(4811)

    loader = DataLoader(
        data, batch_size=params["batch_size"], pin_memory=True, num_workers=num_workers
    )
    trainer = Trainer(
        model, marginal_prob, loader, torch.optim.Adam, params, filepath, device
    )

    trainer.train(params["N_epochs"], early_stopping=50)


def train_model_ddp(
    data,
    model: ScoreModel,
    marginal_prob: MarginalProb,
    params: dict,
    filepath: str,
    device: torch.device,
    num_workers: int = 0,
):
    assert str(device) in [
        "cpu",
        "cuda",
    ], f"{device} is not supported for distributed parallel training"
    ddp_setup(device)
    torch.manual_seed(4811)

    sampler = DistributedSampler(data)
    loader = DataLoader(
        data,
        batch_size=params["batch_size"],
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
    )
    trainer = DDPTrainer(
        model, marginal_prob, loader, torch.optim.Adam, params, filepath, device
    )

    trainer.train(params["N_epochs"], early_stopping=50)

    dist.destroy_process_group()


class Trainer:
    """
    Creates a trainer for a score-based diffusion model.

    Args:
                                                                    _model (torch.nn.Module): the score model class to be trained.
                                                                    _marginal_prob (MarginalProb): marginal probability function of diffusion coefficient.
                                                                    loader (DataLoader): loader of the training data.
                                                                    optimizer (torch.optim.Optimizer): optimisation scheme for training.
                                                                    params (dict): parameters of the model, e.g. batch_size, epochs, hidden layers, channels etc.
                                                                    file_path (str): The path to the file the model state is going to be saved.
                                                                    device (str): The device used to train the model e.g. CPU, GPU, MPS. Defaults to CPUs
    """

    def __init__(
        self,
        _model: torch.nn.Module,
        _marginal_prob: MarginalProb,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        params: dict,
        file_path: str,
        device: torch.device,
    ) -> None:
        self.params = params
        self.file_path = file_path
        self.checkpoint = "checkpoint_" + file_path
        self.loader = loader
        self.epochs = 0
        self.device = device
        self._set_model(_marginal_prob, _model, optimizer)
        self.ckpt_freq = 10

    def _set_model(
        self,
        marginal_prob_cls: MarginalProb,
        score_model: ScoreModel,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        marginal_prob = marginal_prob_cls(sigma=self.params["marginal_prob_sigma"])
        net = Net(
            input_channels=self.params["input_channels"],
            channels=self.params["channels"],
            time_channels=self.params["time_channels"],
            activation=torch.nn.LeakyReLU(inplace=True),
            device=self.device,
        )
        self.model = score_model(
            network=net, marginal_prob=marginal_prob, device=self.device
        )

        if os.path.exists(self.checkpoint):
            self._load_checkpoint(self.checkpoint)

        self.optimizer = optimizer(self.model.parameters(), lr=self.params["base_lr"])

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["MODEL_STATE"])
        self.epochs = ckpt["EPOCHS"]
        if self.rank == 0:
            print("Loading checkpoint")
            print(f"Training continues from checkpoint at epoch {self.epochs}...")

    def _save_checkpoint(self, epoch):
        checkpoint = {"MODEL_STATE": self.model.state_dict(), "EPOCHS": epoch}
        torch.save(checkpoint, self.file_path)
        print(f"Epoch {epoch} | Checkpoint saved at {self.file_path}")

    def train(self, N_epochs: int, eps=1e-5, scheduler=None, early_stopping: int = 10):
        tqdm_epoch = trange(self.epochs, N_epochs)
        best_loss = float("inf")
        counter = 0

        for epoch in tqdm_epoch:
            epoch_loss = 0
            num_items = 0

            for batch in self.loader:
                batch = batch.to(self.device)
                loss = self.model.train_step(batch, self.optimizer, eps, scheduler)
                epoch_loss += loss.item() * batch.shape[0]
                num_items += batch.shape[0]

            current_loss = epoch_loss / num_items
            log_string = f"Average Loss: {current_loss:5f}"

            if epoch % self.ckpt_freq == 0:
                self._save_checkpoint(epoch)

            if best_loss > current_loss:
                counter = 0
                best_loss = current_loss
                torch.save(self.model.state_dict(), self.file_path)
                log_string += " ---> Best model so far (stored)"
            else:
                counter += 1

            tqdm_epoch.set_description(log_string)
            if counter == early_stopping:
                print(
                    f"Stopping training at {epoch:g} epoch(s) ! Best loss: {best_loss: .5f}"
                )
                break


class DDPTrainer(Trainer):
    def __init__(
        self, _model, _marginal_prob, loader, optimizer, params, file_path, device
    ):
        self.rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        super().__init__(
            _model, _marginal_prob, loader, optimizer, params, file_path, device
        )

    def _set_model(self, marginal_prob_cls, score_model, optimizer):
        super()._set_model(marginal_prob_cls, score_model, optimizer)
        self.model = DDP(self.model, device_ids=[self.rank])

    def _save_checkpoint(self, epoch):
        checkpoint = {"MODEL_STATE": self.model.module.state_dict(), "EPOCHS": epoch}
        torch.save(checkpoint, self.file_path)
        print(f"Epoch {epoch} | Checkpoint saved at {self.file_path}")

    def train(self, N_epochs: int, eps=1e-5, scheduler=None, early_stopping: int = 10):
        tqdm_epoch = trange(self.epochs, N_epochs)
        best_loss = float("inf")
        counter = 0

        for epoch in tqdm_epoch:
            epoch_loss = 0
            num_items = 0

            # Sampler shuffles data between epochs
            self.loader.sampler.set_epoch(epoch)

            for batch in self.loader:
                batch = batch.to(self.device)
                loss = self.model.module.train_step(
                    batch, self.optimizer, eps, scheduler
                )

                # Reduce loss across all processes to get avg loss
                reduced_loss = loss.clone()
                dist.reduce(reduced_loss, dst=0, op=dist.ReduceOp.SUM)
                reduced_loss = reduced_loss / self.world_size

                if self.rank == 0:
                    epoch_loss += reduced_loss.item() * batch.shape[0]
                    num_items += batch.shape[0]
            if self.rank == 0:
                current_loss = epoch_loss / num_items
                # self.history.append(current_loss)
                log_string = f"Average Loss: {current_loss:5f}"

                if epoch % self.ckpt_freq == 0:
                    self._save_checkpoint(epoch)

                if best_loss > current_loss:
                    counter = 0
                    best_loss = current_loss
                    torch.save(self.model.module.state_dict(), self.file_path)
                    log_string += " ---> Best model so far (stored)"
                else:
                    counter += 1

                tqdm_epoch.set_description(log_string)
                if counter == early_stopping:
                    print(
                        f"Stopping training at {epoch:g} epoch(s) ! Best loss: {best_loss: .5f}"
                    )
                    break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple score-based diffusion training job"
    )
    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
        help="Max number of epochs to train the model (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Input batch size on device (default: 512)",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Number of workers used by data loader during training (default: 0)",
    )
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--VE", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Load dataset for training
    data = np.loadtxt("data/double_peak_samples_1M.dat", dtype=np.float32)[:, None]

    parameters = {
        "input_channels": 1,
        "marginal_prob_sigma": 10,
        "channels": [64, 64],
        "time_channels": 128,
        "batch_size": args.batch_size,
        "base_lr": 1e-4,
        "N_epochs": args.max_epochs,
    }

    model_filename = (
        f"data/ModelWeights/sigma{parameters['marginal_prob_sigma']}_VE_weights.pt"
    )
    param_filename = (
        f"data/ModelWeights/sigma{parameters['marginal_prob_sigma']}_VE_params.json"
    )
    with open(param_filename, "w") as fp:
        json.dump(parameters, fp)

    device = set_device("gpu" if args.gpu else "cpu")

    if args.ddp and (str(device) in ["cpu", "cuda"]):
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(
                f"Using distributed data parallel training with device: {str(device).upper()}"
            )
        train_func = train_model_ddp
    else:
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(f"{str(device).upper()} is not available for distributed learning.")
        train_func = train_model

    if args.VE:
        model = ScoreModel
        marginal_prob = MarginalProb
    else:
        model = DriftScoreModel
        marginal_prob = VPMarginalProb

    train_func(
        data=data,
        model=model,
        marginal_prob=marginal_prob,
        params=parameters,
        filepath=model_filename,
        device=device,
        num_workers=args.num_workers,
    )
