import os
import torch
from torchvision.utils import save_image
from diffusers.utils import make_image_grid
from diffusers import DDPMPipeline, UNet2DModel
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from PIL import Image

# 设置scheduler的精度为float64
from new_ddpm_scheduler import DDPMScheduler

# 设置全局随机种子函数
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 定义生成图像的函数
@torch.no_grad()
def generate_images(output_dir, num_images=25, image_size=32, grid_size=(5, 5), seed=0):

    set_seed(seed)
    # 加载模型和调度器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = DDPMPipeline.from_pretrained(output_dir).to(device)

    # --------------------float64修改地方-----------------------
    new_noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule="linear",prediction_type="epsilon",variance_type="fixed_large")

    # --------------------float64修改地方-----------------------

    num_inference_steps = 1000
    image_shape = (num_images, pipeline.unet.config.in_channels, image_size, image_size)
    images = torch.randn(image_shape, device=device)
        
    for i, t in tqdm(enumerate(pipeline.scheduler.timesteps), desc=f"generating"):
        with torch.no_grad():
            # import ipdb;ipdb.set_trace();
            model_output = pipeline.unet(images, t).sample
            images = new_noise_scheduler.step(model_output, t, images).prev_sample
            # import ipdb;ipdb.set_trace();
            # 这里采用new_noise_scheduler
        # 在最后一个时间步保存图像
        if (i + 1) == num_inference_steps:
            # import ipdb;ipdb.set_trace();
            generated_images = images / 2 + 0.5  # 将值归一化到 [0, 1]
            generated_images = generated_images.clamp(0, 1).cpu()
            
    # 将 Tensor 图像从 [batch, channel, height, width] 转换为 [height, width, channel]
    generated_images_np = (generated_images.numpy() * 255).astype("uint8")

    # 使用 matplotlib 绘制网格图像
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 1.5, grid_size[0] * 1.5))
    for idx, ax in enumerate(axes.flat):
        if idx < num_images:
            # 转换图像为 [height, width, channels] 格式以供显示
            image = np.transpose(generated_images_np[idx], (1, 2, 0))
            ax.imshow(image)
            ax.axis("off")
        else:
            ax.axis("off")

    # 调整子图间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # 保存网格图像
    save_path = "generated_grid_11_30_float64.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Saved grid image to {save_path}")
                


# 示例用法
output_dir = "ddpm-ema-cifar10-11-30-float64"
generate_images(output_dir, num_images=100, grid_size=(10, 10), seed=0)
