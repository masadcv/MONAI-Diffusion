# %%
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # Using ControlNet to control image generation
#
# This tutorial illustrates how to use MONAI Generative Models to train a ControlNet [1]. ControlNets are hypernetworks that allow for supplying extra conditioning to ready-trained diffusion models. In this example, we will walk through training a ControlNet that allows us to specify a whole-brain mask that the sampled image must respect.
#
#
#
# In summary, the tutorial will cover the following:
# 1. Loading and preprocessing a dataset (we extract the brain MRI dataset 2D slices from 3D volumes from the BraTS dataset)
# 2. Training a 2D diffusion model
# 3. Freeze the diffusion model and train a ControlNet
# 3. Conditional sampling with the ControlNet
#
# [1] - Zhang et al. [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

# %%
# !python -c "import monai" || pip install -q "monai-weekly[tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# %% [markdown]
# ## Setup environment

# %%
import os
import time
import tempfile
import monai
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image

from generative.inferers import ControlNetDiffusionInferer, DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler

print_config()

# %% [markdown]
# ### Setup data directory

# %%
# directory = os.environ.get("MONAI_DATA_DIRECTORY")
# root_dir = tempfile.mkdtemp() if directory is None else directory

# %% [markdown]
# ### Set deterministic training for reproducibility

# %%
set_determinism(42)

# %% [markdown]
# ## Setup BRATS dataset
#
# We now download the BraTS dataset and extract the 2D slices from the 3D volumes.
#

# %% [markdown]
# ### Specify transforms
# We create a rough brain mask by thresholding the image.
output_dir = "./output_multi"
os.makedirs(output_dir, exist_ok=True)

# %%
train_transforms_sl = transforms.Compose(
    [
        transforms.LoadImaged(keys=["source1", "source2", "target"]),
        transforms.EnsureChannelFirstd(keys=["source1", "source2", "target"], channel_dim=-1),
        # transforms.EnsureTyped(keys=["source1", "source2", "target"]),
        # transforms.RandSpatialCropd(keys=["source1", "source2", "target"], roi_size=(64, 64), random_size=False),
        transforms.Resized(
            keys=["source1", "source2", "target"],
            spatial_size=(64, 64),
            mode=("nearest", "nearest", "bilinear"),
        ),
        transforms.ConcatItemsd(keys=["source1", "source2"], name="source", dim=0),
        transforms.NormalizeIntensityd(keys=["source", "target"], divisor=255.0, subtrahend=0),
        transforms.ToTensord(keys=["source", "target"]),
    ]
)

# %% [markdown]
# ### Load training and validation datasets

# %%
# dataset_path = "/data/KCLData/Datasets/How2sign_sample/data_sample_for_ControlNet"
# source_folder_path = os.path.join(dataset_path, "source", "5-rgb")
# target_folder_path = os.path.join(dataset_path, "target", "5-rgb")
dataset_path = "/data/KCLData/Datasets/How2sign_Multi"
source1_folder_path = os.path.join(dataset_path, "keypoints")
# source2_folder_path = os.path.join(dataset_path, "nornal")
source2_folder_path = os.path.join(dataset_path, "normal")
target_folder_path = os.path.join(dataset_path, "frames")
import glob

source1_images_path = glob.glob(os.path.join(source1_folder_path, "*"))
source2_images_path = [
    os.path.join(source2_folder_path, os.path.basename(p)) for p in source1_images_path
]
target_images_path = [
    os.path.join(target_folder_path, os.path.basename(p)) for p in source1_images_path
]

for image in source1_images_path:
    if not os.path.exists(image):
        raise ValueError(f"Image {image} does not exist")
for image in source2_images_path:
    if not os.path.exists(image):
        raise ValueError(f"Image {image} does not exist")
for image in target_images_path:
    if not os.path.exists(image):
        raise ValueError(f"Image {image} does not exist")

list_dict = [
    {"source1": s1, "source2": s2, "target": t} for s1, s2, t in zip(source1_images_path, source2_images_path, target_images_path)
]
train_list = list_dict[: int(len(list_dict) * 0.8)]
val_list = list_dict[int(len(list_dict) * 0.8) :]

# make monai dataset from dictionary of lists
train_ds = monai.data.Dataset(data=train_list, transform=train_transforms_sl)
val_ds = monai.data.Dataset(data=val_list, transform=train_transforms_sl)

# %%
train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    drop_last=True,
    persistent_workers=True,
)

# %% [markdown]
# ### Visualise the images and masks

# %%
check_data = first(train_loader)
print(f"Batch shape: {check_data['target'].shape}")
image_visualisation = torch.cat(
    (
        torch.cat(
            [
                check_data["target"][0],
                check_data["target"][1],
                check_data["target"][2],
                check_data["target"][3],
            ],
            dim=2,
        ),
        torch.cat(
            [
                check_data["source"][0, :3],
                check_data["source"][1, :3],
                check_data["source"][2, :3],
                check_data["source"][3, :3],
            ],
            dim=2,
        ),
        torch.cat(
            [
                check_data["source"][0, 3:],
                check_data["source"][1, 3:],
                check_data["source"][2, 3:],
                check_data["source"][3, 3:],
            ],
            dim=2,
        ),
    ),
    dim=2,
)
image_visualisation = image_visualisation.permute(1, 2, 0).cpu().numpy() * 255
# image_visualisation = image_visualisation.astype("float32") / 255.0
# print(image_visualisation.min())
# print(image_visualisation.max())
# print(image_visualisation.shape)
# plt.figure(figsize=(6, 3))
# # plt.imshow(image_visualisation, vmin=0, vmax=1, cmap="gray")
# plt.imshow(image_visualisation, vmin=0, vmax=1)
# plt.axis("off")
# plt.tight_layout()
# plt.show()
Image.fromarray(image_visualisation.astype(np.uint8)).save(os.path.join(output_dir, "image_visualisation.png"))
# %% [markdown]
# ## Train the Diffusion model
# In general, a ControlNet can be trained in combination with a pre-trained, frozen diffusion model. In this case we will quickly train the diffusion model first.

# %% [markdown]
# ### Define network, scheduler, optimizer, and inferer

# %%
device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256,
)
model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

inferer = DiffusionInferer(scheduler)


# %% [markdown]
# ### Run training
#

# %%
n_epochs = 500
val_interval = 25
epoch_loss_list = []
val_epoch_loss_list = []

scaler = GradScaler()
total_start = time.time()
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0,
                inferer.scheduler.num_train_timesteps,
                (images.shape[0],),
                device=images.device,
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch["target"].to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(
                        0,
                        inferer.scheduler.num_train_timesteps,
                        (images.shape[0],),
                        device=images.device,
                    ).long()
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=model,
                        noise=noise,
                        timesteps=timesteps,
                    )
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

        # Sampling image during training
        noise = torch.randn((1, 3, 64, 64))
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with autocast(enabled=True):
            image = inferer.sample(
                input_noise=noise, diffusion_model=model, scheduler=scheduler
            )

        image = image.permute(0, 2, 3, 1).detach()
        image = np.clip(image.cpu().numpy() * 255, 0, 255)
        image = image.astype(np.uint8)
        batch_dim = image.shape[0]
        sel_batch = np.random.randint(0, batch_dim)
        Image.fromarray(image[sel_batch]).save(os.path.join(output_dir, "dm_training_{}.png".format(epoch)))
        # plt.figure(figsize=(2, 2))
        # plt.imshow(image[0].cpu())  # , vmin=0, vmax=1, cmap="gray")
        # plt.tight_layout()
        # plt.axis("off")
        # plt.show()

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# %% [markdown]
# ## Train the ControlNet

# %% [markdown]
# ### Set up models

# %%
# Create control net
controlnet = ControlNet(
    spatial_dims=2,
    in_channels=3,
    num_channels=(128, 256, 256),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=256,
    conditioning_embedding_num_channels=(16,),
    conditioning_embedding_in_channels=6,
)
# Copy weights from the DM to the controlnet
controlnet.load_state_dict(model.state_dict(), strict=False)
controlnet = controlnet.to(device)
# Now, we freeze the parameters of the diffusion model.
for p in model.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(params=controlnet.parameters(), lr=2.5e-5)
controlnet_inferer = ControlNetDiffusionInferer(scheduler)

# %% [markdown]
# ### Run ControlNet training

# %%
n_epochs = 500
val_interval = 25
epoch_loss_list = []
val_epoch_loss_list = []

scaler = GradScaler()
total_start = time.time()
for epoch in range(n_epochs):
    controlnet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["target"].to(device)
        source = batch["source"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0,
                inferer.scheduler.num_train_timesteps,
                (images.shape[0],),
                device=images.device,
            ).long()

            noise_pred = controlnet_inferer(
                inputs=images,
                diffusion_model=model,
                controlnet=controlnet,
                noise=noise,
                timesteps=timesteps,
                cn_cond=source,
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        controlnet.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch["target"].to(device)
            source = batch["source"].to(device)

            with torch.no_grad():
                with autocast(enabled=True):
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(
                        0,
                        controlnet_inferer.scheduler.num_train_timesteps,
                        (images.shape[0],),
                        device=images.device,
                    ).long()

                    noise_pred = controlnet_inferer(
                        inputs=images,
                        diffusion_model=model,
                        controlnet=controlnet,
                        noise=noise,
                        timesteps=timesteps,
                        cn_cond=source,
                    )
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()

            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            break

        val_epoch_loss_list.append(val_epoch_loss / (step + 1))

        # Sampling image during training with controlnet conditioning

        with torch.no_grad():
            with autocast(enabled=True):
                noise = torch.randn((1, 3, 64, 64)).to(device)
                sample = controlnet_inferer.sample(
                    input_noise=noise,
                    diffusion_model=model,
                    controlnet=controlnet,
                    cn_cond=source[0, None, ...],
                    scheduler=scheduler,
                )

        # Without using an inferer:
        #         progress_bar_sampling = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=110)
        #         progress_bar_sampling.set_description("sampling...")
        #         sample = torch.randn((1, 1, 64, 64)).to(device)
        #         for t in progress_bar_sampling:
        #             with torch.no_grad():
        #                 with autocast(enabled=True):
        #                     down_block_res_samples, mid_block_res_sample = controlnet(
        #                         x=sample, timesteps=torch.Tensor((t,)).to(device).long(), controlnet_cond=source[0, None, ...]
        #                     )
        #                     noise_pred = model(
        #                         sample,
        #                         timesteps=torch.Tensor((t,)).to(device),
        #                         down_block_additional_residuals=down_block_res_samples,
        #                         mid_block_additional_residual=mid_block_res_sample,
        #                     )
        #                     sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)
        samples = sample.permute(0, 2, 3, 1).detach()
        source = source.permute(0, 2, 3, 1).detach()
        # plt.subplots(1, 2, figsize=(4, 2))
        # plt.subplot(1, 2, 1)
        # plt.imshow(source[0].cpu(), vmin=0, vmax=1)
        # # plt.imshow(source[0].cpu(), vmin=0, vmax=1, cmap="gray")
        # plt.axis("off")
        # plt.title("Conditioning mask")
        # plt.subplot(1, 2, 2)
        # plt.imshow(sample[0].cpu(), vmin=0, vmax=1)
        # # plt.imshow(sample[0].cpu(), vmin=0, vmax=1, cmap="gray")
        # plt.axis("off")
        # plt.title("Sample image")
        # plt.tight_layout()
        # plt.axis("off")
        # plt.show()
        batch_dim = samples.shape[0]
        sel_batch = np.random.randint(0, batch_dim)
        samples = np.clip(samples.cpu().numpy() * 255, 0, 255)
        samples = samples.astype(np.uint8)
        Image.fromarray(samples[sel_batch]).save(os.path.join(output_dir, "cn_training_sample_{}.png".format(epoch)))

        source = np.clip(source.cpu().numpy() * 255, 0, 255)
        source = source.astype(np.uint8)
        source1 = source[:, :, :, :3]
        Image.fromarray(source1[sel_batch]).save(os.path.join(output_dir, "cn_training_source1_{}.png".format(epoch)))
        source2 = source[:, :, :, 3:]
        Image.fromarray(source2[sel_batch]).save(os.path.join(output_dir, "cn_training_source2_{}.png".format(epoch)))

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")

# %% [markdown]
# ## Sample with ControlNet conditioning
# First we'll provide a few different masks from the validation data as conditioning. The samples should respect the shape of the conditioning mask, but don't need to have the same content as the corresponding validation image.

# %%
progress_bar_sampling = tqdm(
    scheduler.timesteps,
    total=len(scheduler.timesteps),
    ncols=110,
    position=0,
    leave=True,
)
progress_bar_sampling.set_description("sampling...")
num_samples = 4
sample = torch.randn((num_samples, 3, 64, 64)).to(device)

val_batch = first(val_loader)
val_images = val_batch["target"].to(device)
val_source = val_batch["source"].to(device)
for t in progress_bar_sampling:
    with torch.no_grad():
        with autocast(enabled=True):
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=sample,
                timesteps=torch.Tensor((t,)).to(device).long(),
                controlnet_cond=val_source[:num_samples, ...],
            )
            noise_pred = model(
                sample,
                timesteps=torch.Tensor((t,)).to(device),
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
            sample, _ = scheduler.step(
                model_output=noise_pred, timestep=t, sample=sample
            )
val_source = val_source.permute(0, 2, 3, 1).detach()
val_images = val_images.permute(0, 2, 3, 1).detach()
sample = sample.permute(0, 2, 3, 1).detach()
# plt.subplots(num_samples, 3, figsize=(6, 8))
# for k in range(num_samples):
#     plt.subplot(num_samples, 3, k * 3 + 1)
#     plt.imshow(val_source[k, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
#     plt.axis("off")
#     if k == 0:
#         plt.title("Conditioning mask")
#     plt.subplot(num_samples, 3, k * 3 + 2)
#     plt.imshow(val_images[k, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
#     plt.axis("off")
#     if k == 0:
#         plt.title("Actual val image")
#     plt.subplot(num_samples, 3, k * 3 + 3)
#     plt.imshow(sample[k, 0, ...].cpu(), vmin=0, vmax=1, cmap="gray")
#     plt.axis("off")
#     if k == 0:
#         plt.title("Sampled image")
# plt.tight_layout()
# plt.show()
val_source = np.clip(val_source.cpu().numpy() * 255, 0, 255)
val_source = val_source.astype(np.uint8)
val_source1 = val_source[:, :, :, :3]
val_source2 = val_source[:, :, :, 3:]

val_images = np.clip(val_images.cpu().numpy() * 255, 0, 255)
val_images = val_images.astype(np.uint8)

sample = np.clip(sample.cpu().numpy() * 255, 0, 255)
sample = sample.astype(np.uint8)

for k in range(num_samples):
    Image.fromarray(val_images[k]).save(os.path.join(output_dir, "cn_val_image_{}.png".format(k)))
    Image.fromarray(sample[k]).save(os.path.join(output_dir, "cn_val_sample_{}.png".format(k)))
    Image.fromarray(val_source1[k]).save(os.path.join(output_dir, "cn_val_source1_{}.png".format(k)))
    Image.fromarray(val_source2[k]).save(os.path.join(output_dir, "cn_val_source2_{}.png".format(k)))