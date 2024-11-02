from math import log
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm, trange
from utils import grab


class GaussianFourierProjection(torch.nn.Module):
    def __init__(
        self, embed_dim: int, scale: float = 30.0, device: Optional[str] = None
    ) -> None:
        super().__init__()
        # self.W = torch.nn.Parameter(torch.randn(embed_dim // 2, device=device) * scale, requires_grad=False)
        self.scale = log(10_000) / (embed_dim // 2)
        self.W = torch.exp(torch.arange(embed_dim // 2, device=device) * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :]  # * 2 * torch.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)


class ConditionalLinear(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        activation: torch.nn.Module = torch.nn.SiLU(),
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=in_channels, out_features=out_channels, device=device
        )
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embed_dim=time_channels, device=device),
            torch.nn.Linear(time_channels, out_channels, device=device),
        )
        self.act = activation

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.linear(x)
        x = x + self.embed(t)
        return self.act(x)


class Net(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        channels: list = [64, 64],
        time_channels: int = 64,
        activation: torch.nn.Module = torch.nn.SiLU(),
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Network architecture layers
        self.channels = [input_channels] + channels
        self.hidden_layers = torch.nn.ModuleList(
            [
                ConditionalLinear(
                    self.channels[i],
                    self.channels[i + 1],
                    time_channels,
                    activation=activation,
                    device=device,
                )
                for i in range(len(self.channels) - 1)
            ]
        )
        self.final = torch.nn.Linear(channels[-1], input_channels, device=device)
        # Model's parameter

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x, t)
        return self.final(x)


class MarginalProb:
    def __init__(self, sigma: float = 2.0) -> None:
        self.logsigma = log(sigma)

    def get_mean_stddev(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (
            x,
            torch.sqrt((torch.exp(2 * t * self.logsigma) - 1.0) / 2 / self.logsigma)[
                ..., None
            ],
        )

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(t * self.logsigma)[..., None]

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class VPMarginalProb(MarginalProb):
    def __init__(self, sigma: float = 2) -> None:
        super().__init__(sigma)

    def get_mean_stddev(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = ((torch.exp(2 * t * self.logsigma) - 1.0) / 2 / self.logsigma)[..., None]
        return x * torch.exp(-s / 2), torch.sqrt(1 - torch.exp(-s))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.diffusion_coeff(t) ** 2 * x


class ScoreModel(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        marginal_prob: MarginalProb,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.marginal_prob = marginal_prob
        self.device = device
        self.history = []

    def forward(self, x, t):
        return self.network(x, t)

    def train_step(self, x, optimizer, eps, scheduler=None):
        random_t = torch.rand(1, device=self.device) * (1.0 - eps) + eps
        z = torch.randn_like(x)
        mean, std = self.marginal_prob.get_mean_stddev(x, random_t)
        perturbed_x = mean + z * std

        optimizer.zero_grad()

        score = self.forward(perturbed_x, random_t)
        loss = 0.5 * torch.mean(torch.sum((score * std + z) ** 2, dim=-1))

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return loss

    def train(
        self,
        loader,
        optimizer,
        epochs: int,
        filename: str,
        eps: float = 1e-5,
        scheduler=None,
        early_stopping: int = 10,
    ) -> None:
        tqdm_epoch = trange(epochs)
        best_loss = float("inf")
        counter = 0

        for epoch in tqdm_epoch:
            epoch_loss = 0
            num_items = 0
            for batch in loader:
                batch = torch.concatenate([batch, -batch])
                loss = self.train_step(batch.to(self.device), optimizer, eps, scheduler)
                epoch_loss += loss.item() * batch.shape[0]
                num_items += batch.shape[0]
            current_loss = epoch_loss / num_items
            self.history.append(current_loss)
            log_string = f"Average Loss: {self.history[-1]:5f}"

            if best_loss > current_loss:
                counter = 0
                best_loss = current_loss
                torch.save(self.state_dict(), filename)
                log_string += " ---> Best model so far (stored)"

            counter += 1
            tqdm_epoch.set_description(log_string)
            if counter == early_stopping:
                print(
                    f"Stopping training at {epoch:g} epoch(s) ! Best loss: {best_loss: .5f}"
                )
                break

    def sampler(self, size: int, num_steps: int, history: bool = False, eps=1e-5):
        output = []
        _, std = self.marginal_prob.get_mean_stddev(
            None, torch.ones(1)
        )  # we only need the std. dev.
        x = np.random.normal(loc=0, scale=std.item(), size=(size, 1))
        step_size = 1 / num_steps
        step_size_sqrt = step_size**0.5

        with torch.no_grad():
            for t_i in tqdm(torch.linspace(1, eps, num_steps)[:, None]):
                g_t = self.marginal_prob.diffusion_coeff(t_i).item()

                torch_x = torch.tensor(x, dtype=torch.float32)
                score = grab(self.forward(torch_x, t_i))

                noise = np.random.randn(*x.shape) if t_i > eps else 0

                x = (x + step_size * g_t**2 * score) + step_size_sqrt * g_t * noise
                if history:
                    output.append(x)

            if history:
                return np.stack(output)
        return x

    def tensor_sampler(
        self, size: int, num_steps: int, history: bool = False, eps=1e-5
    ):
        output = []
        step_size = 1 / num_steps
        step_size_sqrt = step_size**0.5

        t = torch.ones(1, device=self.device)
        _, std = self.marginal_prob.get_mean_stddev(None, t)
        x = torch.randn(size, 1, device=self.device) * std
        with torch.no_grad():
            for t_i in tqdm(
                torch.linspace(1, eps, num_steps, device=self.device)[:, None]
            ):
                g_t = self.marginal_prob.diffusion_coeff(t_i)
                noise = torch.randn_like(x) if t_i > eps else 0

                x = (
                    x
                    + step_size * g_t**2 * self.forward(x, t_i)
                    + step_size_sqrt * g_t * noise
                )
                if history:
                    output.append(x)
            if history:
                return torch.stack(output)
        return x


class DriftScoreModel(ScoreModel):
    def __init__(
        self,
        network: torch.nn.Module,
        marginal_prob: VPMarginalProb,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(network, marginal_prob, device)

    def train_step(self, x, optimizer, eps, scheduler=None):
        random_t = torch.rand(1, device=self.device) * (1.0 - eps) + eps
        z = torch.randn_like(x)
        mean, std = self.marginal_prob.get_mean_stddev(x, random_t)
        perturbed_x = mean + z * std

        optimizer.zero_grad()

        score = self.forward(perturbed_x, random_t)
        loss = 0.5 * torch.mean(torch.sum((score * std + z) ** 2, dim=-1))

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return loss

    def sampler(
        self, size: int, num_steps: int, history: bool = False, eps=1e-5, init_x=None
    ):
        x = np.random.normal(0, 1, size=(size, 1)) if init_x is None else init_x
        output = [x]
        step_size = 1 / num_steps
        step_size_sqrt = step_size**0.5

        for t_i in tqdm(torch.linspace(1, eps, num_steps)[:, None]):
            g_t = self.marginal_prob.diffusion_coeff(t_i).item()

            torch_x = torch.tensor(x, dtype=torch.float32)
            score = grab(self.forward(torch_x, t_i))
            drift = grab(self.marginal_prob.drift(torch_x, t_i))

            noise = np.random.randn(*x.shape) if t_i > eps else 0

            x = (
                x
                + step_size * (-drift + g_t**2 * score)
                + step_size_sqrt * g_t * noise
            )
            if history:
                output.append(x)

        if history:
            return np.stack(output)
        return x

    def tensor_sampler(
        self, size: int, num_steps: int, history: bool = False, eps=1e-5
    ):
        output = []
        step_size = 1 / num_steps
        step_size_sqrt = step_size**0.5

        x = torch.randn(size, 1, device=self.device)
        with torch.no_grad():
            for t_i in tqdm(
                torch.linspace(1, eps, num_steps, device=self.device)[:, None]
            ):
                g_t = self.marginal_prob.diffusion_coeff(t_i)
                noise = torch.randn_like(x) if t_i > eps else 0
                drift = self.marginal_prob.drift(x, t_i)
                score = self.forward(x, t_i)

                x = (
                    x
                    + step_size * (-drift + g_t**2 * score)
                    + step_size_sqrt * g_t * noise
                )
                if history:
                    output.append(x)
            if history:
                return torch.stack(output)
        return x
