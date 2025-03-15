import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import random

from transformers import T5Tokenizer
import wandb

from src.dataloaders.dataloader import TextImageDataset
from src.models.chroma.model import Chroma, chroma_params
from src.models.chroma.utils import (
    vae_flatten,
    prepare_latent_image_ids,
    vae_unflatten,
    calculate_shift,
    time_shift,
)
from src.models.chroma.module.autoencoder import AutoEncoder, ae_params
from src.math_utils import cosine_optimal_transport
from src.models.chroma.module.t5 import T5EncoderModel, T5Config, replace_keys
from src.general_utils import load_file_multipart, load_selected_keys, load_safetensors
from src.lora_and_quant import (
    swap_linear_simple,
    LinearWithLoRA,
    Quantized4BitLinearWithLoRA,
    Quantized8BitLinearWithLoRA,
    find_lora_params,
)

from huggingface_hub import HfApi, upload_file
import time


@dataclass
class TrainingConfig:
    total_epochs: int
    master_seed: int
    train_minibatch: int
    lr: float
    weight_decay: float
    save_folder: str
    wandb_key: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    wandb_entity: Optional[str] = None
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None


@dataclass
class DataloaderConfig:
    batch_size: int
    jsonl_metadata_path: str
    image_folder_path: str
    base_resolution: list[int]
    shuffle_tags: bool
    tag_drop_percentage: float
    uncond_percentage: float
    resolution_step: int
    num_workers: int
    prefetch_factor: int
    ratio_cutoff: float
    thread_per_worker: int


@dataclass
class ModelConfig:
    """Dataclass to store model paths."""

    chroma_path: str
    vae_path: str
    t5_path: str
    t5_config_path: str
    t5_tokenizer_path: str
    t5_to_8bit: bool
    t5_max_length: int


@dataclass
class LoraConfig:
    rank: int
    alpha: int
    target_layers: list[str]
    base_model_quant_level: Optional[str] = "full"


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def create_distribution(num_points, device=None):
    # Probability range on x axis
    x = torch.linspace(0, 1, num_points, device=device)

    # Custom probability density function
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2

    # Normalize to sum to 1
    probabilities /= probabilities.sum()

    return x, probabilities


# Upload the model to Hugging Face Hub
def upload_to_hf(model_filename, path_in_repo, repo_id, token, max_retries=3):
    api = HfApi()

    for attempt in range(max_retries):
        try:
            upload_file(
                path_or_fileobj=model_filename,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
            )
            print(f"Model uploaded to {repo_id}/{path_in_repo}")
            return  # Exit function if successful

        except Exception as e:
            print(f"Upload attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)  # Exponential backoff

    print("Upload failed after multiple attempts.")


def sample_from_distribution(x, probabilities, num_samples, device=None):
    # Step 1: Compute the cumulative distribution function
    cdf = torch.cumsum(probabilities, dim=0)

    # Step 2: Generate uniform random samples
    uniform_samples = torch.rand(num_samples, device=device)

    # Step 3: Map uniform samples to the x values using the CDF
    indices = torch.searchsorted(cdf, uniform_samples, right=True)

    # Get the corresponding x values for the sampled indices
    sampled_values = x[indices]

    return sampled_values


def prepare_sot_pairings(latents):
    # stochastic optimal transport pairings
    # just use mean because STD is so small and practically negligible
    latents = latents.to(torch.float32)
    latents, latent_shape = vae_flatten(latents)
    n, c, h, w = latent_shape
    image_pos_id = prepare_latent_image_ids(n, h, w)

    # randomize ode timesteps
    # input_timestep = torch.round(
    #     F.sigmoid(torch.randn((n,), device=latents.device)), decimals=3
    # )
    num_points = 1000  # Number of points in the range
    x, probabilities = create_distribution(num_points, device=latents.device)
    input_timestep = sample_from_distribution(
        x, probabilities, n, device=latents.device
    )

    # biasing towards earlier more noisy steps where it's the most uncertain
    input_timestep = time_shift(0.5, 1, input_timestep)

    timesteps = input_timestep[:, None, None]
    # 1 is full noise 0 is full image
    noise = torch.randn_like(latents)

    # compute OT pairings
    transport_cost, indices = cosine_optimal_transport(
        latents.reshape(n, -1), noise.reshape(n, -1)
    )
    noise = noise[indices[1].view(-1)]

    # random lerp points
    noisy_latents = latents * (1 - timesteps) + noise * timesteps

    # target vector that being regressed on
    target = noise - latents

    return noisy_latents, target, input_timestep, image_pos_id, latent_shape


def init_optimizer(model, trained_layer_keywords, lr, wd, t_total):
    """Initialize AdamW (8-bit) optimizer with CosineAnnealingWarmRestarts scheduler."""
    trained_params = []
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trained_layer_keywords):
            param.requires_grad = True
            trained_params.append(param)
        else:
            param.requires_grad = False

    optimizer = bnb.optim.AdamW8bit(
        trained_params,
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(t_total // 10, 1),  # First restart after 10% of training steps
        T_mult=2,  # Double the restart period each time
        eta_min=lr * 0.1,  # Minimum learning rate
    )

    return optimizer, scheduler


def synchronize_gradients(model, scale=1):
    for param in model.parameters():
        if param.grad is not None:
            # Synchronize gradients by summing across all processes
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Average the gradients if needed
            if scale > 1:
                param.grad /= scale


def save_part(model, trained_layer_keywords, path):
    full_state_dict = model.state_dict()

    filtered_state_dict = {}
    for k, v in full_state_dict.items():
        if any(keyword in k for keyword in trained_layer_keywords):
            filtered_state_dict[k] = v

    torch.save(filtered_state_dict, path)


def cast_linear(module, dtype):
    """
    Recursively cast all nn.Linear layers in the model to bfloat16.
    """
    for name, child in module.named_children():
        # If the child module is nn.Linear, cast it to bf16
        if isinstance(child, nn.Linear):
            child.to(dtype)
        else:
            # Recursively apply to child modules
            cast_linear(child, dtype)


def save_config_to_json(filepath: str, **configs):
    json_data = {key: asdict(value) for key, value in configs.items()}
    with open(filepath, "w") as json_file:
        json.dump(json_data, json_file, indent=4)


def dump_dict_to_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_config_from_json(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def cache_latents(dataset, model_config, rank, batch_size):
    """Precompute and store batched latents and embeddings before training to save VRAM."""
    latents_cache = []
    embeddings_cache = []
    masks_cache = []

    with torch.no_grad():
        ae = AutoEncoder(ae_params).to(rank)
        t5 = T5EncoderModel(T5Config.from_json_file(model_config.t5_config_path)).to(rank)
        t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)

        acc_latents = []
        acc_embeddings = []
        acc_masks = []

        for images, captions, _ in tqdm(dataset, desc="Caching latents and embeddings"):
            text_inputs = t5_tokenizer(
                captions,
                padding="max_length",
                max_length=model_config.t5_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(rank)

            latents = ae.encode_for_train(images.to(rank)).to("cpu")
            embeddings = t5(text_inputs.input_ids, text_inputs.attention_mask).to("cpu")
            masks = text_inputs.attention_mask.to("cpu")

            acc_latents.append(latents)
            acc_embeddings.append(embeddings)
            acc_masks.append(masks)

            if len(acc_latents) >= batch_size:
                latents_cache.append(torch.cat(acc_latents, dim=0))
                embeddings_cache.append(torch.cat(acc_embeddings, dim=0))
                masks_cache.append(torch.cat(acc_masks, dim=0))
                acc_latents, acc_embeddings, acc_masks = [], [], []

        if acc_latents:  # Add remaining data
            latents_cache.append(torch.cat(acc_latents, dim=0))
            embeddings_cache.append(torch.cat(acc_embeddings, dim=0))
            masks_cache.append(torch.cat(acc_masks, dim=0))

        # Offload to CPU
        ae.to("cpu")
        t5.to("cpu")
        torch.cuda.empty_cache()

    return latents_cache, embeddings_cache, masks_cache


def train_chroma(rank, world_size, debug=False):
    # Initialize distributed training
    if not debug:
        setup_distributed(rank, world_size)

    config_data = load_config_from_json("training_config_chroma_lora.json")

    training_config = TrainingConfig(**config_data["training"])
    dataloader_config = DataloaderConfig(**config_data["dataloader"])
    model_config = ModelConfig(**config_data["model"])
    lora_config = LoraConfig(**config_data["lora"])

    # wandb logging
    if training_config.wandb_project is not None and rank == 0:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if training_config.wandb_key:
            wandb.login(key=training_config.wandb_key)
        wandb.init(
            project=training_config.wandb_project,
            name=f"{training_config.wandb_run}_{current_datetime}",
            entity=training_config.wandb_entity,
        )

    os.makedirs(training_config.save_folder, exist_ok=True)
    # paste the training config for this run
    dump_dict_to_json(
        config_data, f"{training_config.save_folder}/training_config.json"
    )

    # global training RNG
    torch.manual_seed(training_config.master_seed)
    random.seed(training_config.master_seed)

    # load model
    with torch.no_grad():
        # load chroma and enable grad
        chroma_params._use_compiled = True
        with torch.device("meta"):
            model = Chroma(chroma_params)
        model.load_state_dict(load_safetensors(model_config.chroma_path), assign=True)
        model.to(torch.bfloat16)
        model.to(rank)

        # set trainable lora layer
        lora_module = {
            "full": LinearWithLoRA,
            "8bit": Quantized8BitLinearWithLoRA,
            "4bit": Quantized4BitLinearWithLoRA,
        }

        swap_linear_simple(
            model,
            lora_module[lora_config.base_model_quant_level],
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            include_keywords=lora_config.target_layers,
        )

        trained_layer_keywords = []
        for n, p in find_lora_params(model):
            trained_layer_keywords.append(n)
            p.data = p.data.to(torch.bfloat16)

        # load ae
        with torch.device("meta"):
            ae = AutoEncoder(ae_params)
        ae.load_state_dict(load_safetensors(model_config.vae_path), assign=True)
        ae.to(torch.bfloat16)

        # load t5
        t5_tokenizer = T5Tokenizer.from_pretrained(model_config.t5_tokenizer_path)
        t5_config = T5Config.from_json_file(model_config.t5_config_path)
        with torch.device("meta"):
            t5 = T5EncoderModel(t5_config)
        t5.load_state_dict(
            replace_keys(load_file_multipart(model_config.t5_path)), assign=True
        )
        t5.eval()
        t5.to(torch.bfloat16)
        if model_config.t5_to_8bit:
            cast_linear(t5, torch.float8_e4m3fn)

    dataset = TextImageDataset(
        batch_size=dataloader_config.batch_size,
        jsonl_path=dataloader_config.jsonl_metadata_path,
        image_folder_path=dataloader_config.image_folder_path,
        # don't use this tag implication pruning it's slow!
        # preprocess the jsonl tags before training!
        # tag_implication_path="tag_implications.csv",
        base_res=dataloader_config.base_resolution,
        shuffle_tags=dataloader_config.shuffle_tags,
        tag_drop_percentage=dataloader_config.tag_drop_percentage,
        uncond_percentage=dataloader_config.uncond_percentage,
        resolution_step=dataloader_config.resolution_step,
        seed=training_config.master_seed,
        rank=rank,
        num_gpus=world_size,
        ratio_cutoff=dataloader_config.ratio_cutoff,
    )

    # Cache latents and embeddings before training
    latents_cache, embeddings_cache, masks_cache = cache_latents(dataset, model_config, rank, training_config.train_minibatch)

    model.model.requires_grad_(True)
    scaler = torch.cuda.amp.GradScaler()
    optimizer, scheduler = init_optimizer(
        model,
        trained_layer_keywords,
        training_config.lr,
        training_config.weight_decay,
        training_config.total_epochs*len(latents_cache),  # Total training steps
    )

    for i in range(0, training_config.total_epochs):
        training_config.master_seed += 1
        torch.manual_seed(training_config.master_seed)
        for counter, (latents, embeddings, masks) in enumerate(zip(latents_cache, embeddings_cache, masks_cache)):
            latents, embeddings, masks = latents.to(rank), embeddings.to(rank), masks.to(rank)

            if not debug:
                dist.barrier()

            # move model to device
            model.to(rank)

            #acc_latents = torch.cat(acc_latents, dim=0)
            #acc_embeddings = torch.cat(acc_embeddings, dim=0)
            #acc_mask = torch.cat(acc_mask, dim=0)
            acc_latents = latents
            acc_embeddings = embeddings
            acc_mask = masks

            # prepare flat image and the target lerp
            (
                noisy_latents,
                target,
                input_timestep,
                image_pos_id,
                latent_shape,
            ) = prepare_sot_pairings(acc_latents.to(rank))
            noisy_latents = noisy_latents.to(torch.bfloat16)
            target = target.to(torch.bfloat16)
            input_timestep = input_timestep.to(torch.bfloat16)
            image_pos_id = image_pos_id.to(rank)

            # t5 text id for the model
            text_ids = torch.zeros((noisy_latents.shape[0], 512, 3), device=rank)
            # NOTE:
            # using static guidance 1 for now
            # this should be disabled later on !
            static_guidance = torch.tensor(
                [0.0] * acc_latents.shape[0], device=rank
            )

            # set the input to requires grad to make autograd works
            noisy_latents.requires_grad_(True)
            acc_embeddings.requires_grad_(True)

            # aliasing
            optimizer.zero_grad()

            mb = training_config.train_minibatch
            loss_log = []
            for tmb_i in tqdm(
                range(dataloader_config.batch_size // mb // world_size),
                desc=f"minibatch training, Rank {rank}",
                position=rank,
            ):
                # do this inside for loops!
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(
                        img=noisy_latents[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        img_ids=image_pos_id[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt=acc_embeddings[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt_ids=text_ids[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        txt_mask=acc_mask[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        timesteps=input_timestep[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                        guidance=static_guidance[tmb_i * mb : tmb_i * mb + mb].to(
                            rank, non_blocking=True
                        ),
                    )
                    # TODO: need to scale the loss with rank count and grad accum!
                    loss = F.mse_loss(
                        pred,
                        target[tmb_i * mb : tmb_i * mb + mb],
                    ) / (dataloader_config.batch_size // mb)

                scaler.scale(loss).backward()

                loss_log.append(
                    loss.detach().clone() * (dataloader_config.batch_size // mb)
                )
            loss_log = sum(loss_log) / len(loss_log)

            if not debug:
                synchronize_gradients(model)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if training_config.wandb_project is not None and rank == 0:
                wandb.log({"loss": loss_log, "lr": training_config.lr})

            if not debug:
                dist.barrier()


        # save final model
        model_filename = f"{training_config.save_folder}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        save_part(model, trained_layer_keywords, model_filename)
        if training_config.hf_token:
            upload_to_hf(
                model_filename,
                model_filename,
                training_config.hf_repo_id,
                training_config.hf_token,
            )

    if not debug:
        dist.destroy_process_group()
