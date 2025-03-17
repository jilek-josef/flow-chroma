import torch
from torch.optim import AdamW
from collections import defaultdict

from torch.optim.adamw import adamw


class LoliAdamW(AdamW):
    def __init__(
        self,
        params,
        num_clusters=10,  # Number of optimizer clusters
        max_timesteps=1000,  # Maximum diffusion timestep
        **kwargs
    ):
        """
        Loli AdamW: Treats different timesteps independently by maintaining
        separate optimizer states for different timestep clusters.

        Args:
            params: Model parameters
            num_clusters (int): Number of timestep clusters (fewer clusters = less memory)
            max_timesteps (int): Maximum diffusion timestep in the dataset
            **kwargs: Other AdamW arguments (e.g., lr, betas, weight_decay)
        """
        super().__init__(params, **kwargs)

        self.num_clusters = num_clusters
        self.cluster_size = max_timesteps // num_clusters  # Timesteps per cluster

        # Create a dictionary to hold optimizer states for each cluster
        self.cluster_states = defaultdict(lambda: defaultdict(dict))

    def get_cluster_idx(self, timestep):
        """Assign timestep to a cluster index."""
        print(min(timestep // self.cluster_size, self.num_clusters - 1))
        return min(timestep // self.cluster_size, self.num_clusters - 1)

    def step(self, closure=None, timestep=None):
        """
        Performs a single optimization step with per-cluster optimizer states.

        Args:
            closure (optional): A closure that reevaluates the model and returns the loss.
            timestep (float): Current diffusion timestep (0 to 1) -> will be multiplied to match timestep
        """
        if timestep is None:
            raise ValueError("Timestep must be provided for LoliAdamW.")
        timestep = (torch.tensor(timestep) * 1000).mean().round().long()
        print(timestep)
        loss = None
        if closure is not None:
            loss = closure()

        # Determine cluster index based on the timestep
        cluster_idx = self.get_cluster_idx(timestep)

        # Get optimizer state for this cluster
        state_dict = self.cluster_states[cluster_idx]

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            # Use the optimizer state from the correct cluster
            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                # Get or initialize cluster-specific optimizer state
                state = state_dict[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            # Perform the standard AdamW update using the selected cluster state
            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs if amsgrad else None,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
            )

        return loss