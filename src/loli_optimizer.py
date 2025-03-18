import torch
from numpy.ma.core import append
from torch.optim import AdamW, Optimizer


class LoliOptimizer:
    def __init__(self, num_clusters, max_timesteps, lr_scheduler, scheduler_kwargs, optimizer_kwargs):
        """
        Interface for Loli optimizers that maintain separate optimizer states per timestep cluster.

        Args:
            num_clusters (int): Number of timestep clusters
            max_timesteps (int): Maximum diffusion timestep
        """
        self.lr_scheduler_states = None #must be implemented
        self.cluster_states = None #must be implemented
        self.num_clusters = num_clusters
        self.cluster_size = max_timesteps // num_clusters
        self.lr_scheduler = lr_scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs

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
            raise ValueError("Timestep must be provided for LoliOptimizer.")
        timestep = (torch.tensor(timestep) * 1000).mean().round().long()

        cluster_id = self.get_cluster_idx(timestep)
        optimizer_step = self.cluster_states[cluster_id].step(closure=closure)
        self.lr_scheduler_states[cluster_id].step()

        return optimizer_step

    def zero_grad(self, timestep=None):
        if timestep is None:
            raise ValueError("Timestep must be provided for LoliOptimizer.")
        timestep = (torch.tensor(timestep) * 1000).mean().round().long()

        cluster_id = self.get_cluster_idx(timestep)
        self.cluster_states[cluster_id].zero_grad()


class LoliAdamW(LoliOptimizer):
    def __init__(self,
                 params,
                 num_clusters=10,
                 max_timesteps=1000,
                 lr_scheduler="CosineAnnealingWarmRestarts",
                 scheduler_kwargs=None,
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
            scheduler_kwargs ({}): Mandatory, Scheduler arguments
            optimizer_kwargs ({}): Mandatory, AdamW arguments (e.g., lr, betas, weight_decay)
        """

        super().__init__(num_clusters, max_timesteps, lr_scheduler, scheduler_kwargs, optimizer_kwargs)

        # Create a dictionary to hold optimizer states for each cluster
        self.cluster_states = list()
        self.lr_scheduler_states = list()
        for i in range(num_clusters):
            optimizer = AdamW(params=params, **optimizer_kwargs)
            if self.lr_scheduler == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    **scheduler_kwargs
                )
                self.lr_scheduler_states.append(scheduler)
            self.cluster_states.append(optimizer)
