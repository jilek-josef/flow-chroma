import torch
from numpy.ma.core import append
from torch.optim import AdamW, Optimizer


class LoliAdamW(Optimizer):
    def __init__(
        self,
        params,
        num_clusters=10,  # Number of optimizer clusters
        max_timesteps=1000,  # Maximum diffusion timestep
        lr_scheduler="CosineAnnealingWarmRestarts",
        scheduler_T_0=100,
        scheduler_T_mult=2,
        scheduler_eta_min=1e-6,
        scheduler_last_epoch=-1,
        optimizer_kwargs=None
    ):
        """
        Loli AdamW: Treats different timesteps independently by maintaining
        separate optimizer states for different timestep clusters.

        Args:
            params: Model parameters
            num_clusters (int): Number of timestep clusters (fewer clusters = less memory)
            max_timesteps (int): Maximum diffusion timestep in the dataset
            lr_scheduler="CosineAnnealingWarmRestarts": currently no other supported
            scheduler_T_0 (int) Scheduler parameters here
            scheduler_T_mult (int)
            scheduler_eta_min (float)
            **kwargs: Other AdamW arguments (e.g., lr, betas, weight_decay)
        """
        super().__init__(params, **optimizer_kwargs)

        self.num_clusters = num_clusters
        self.cluster_size = max_timesteps // num_clusters  # Timesteps per cluster
        self.lr_scheduler = "CosineAnnealingWarmRestarts"

        # Create a dictionary to hold optimizer states for each cluster
        self.cluster_states = list()
        self.lr_scheduler_states = list()
        for i in range(num_clusters):
            optimizer = AdamW(params=params, **optimizer_kwargs)
            if lr_scheduler == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=scheduler_T_0,
                    T_mult=scheduler_T_mult,
                    eta_min=scheduler_eta_min,
                    last_epoch=scheduler_last_epoch
                )
                self.lr_scheduler_states.append(scheduler)
            self.cluster_states.append(optimizer)


    def get_cluster_idx(self, timestep):
        """Assign timestep to a cluster index."""
        return min(timestep // self.cluster_size, self.num_clusters - 1)

    def step(self, closure=None, timestep=None):
        """
        Performs a single optimization step with per-cluster optimizer states. Also performs a single scheduler step of the respective optimizer.

        Args:
            closure (optional): A closure that reevaluates the model and returns the loss.
            timestep (float): Current diffusion timestep (0 to 1) -> will be multiplied to match timestep
        """
        if timestep is None:
            raise ValueError("Timestep must be provided for LoliAdamW.")
        timestep = (torch.tensor(timestep) * 1000).mean().round().long()

        cluster_id = self.get_cluster_idx(timestep)
        optimizer_step = self.cluster_states[cluster_id].step(closure=closure)
        self.lr_scheduler_states[cluster_id].step()

        return optimizer_step