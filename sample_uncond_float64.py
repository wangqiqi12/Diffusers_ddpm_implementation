import os
import torch
from diffusers import DDPMPipeline
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
import argparse

# 设置scheduler的精度为float64
from new_ddpm_scheduler import DDPMScheduler

def evaluate(config):
    # 检查输出目录是否存在
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    # 加载训练好的模型
    print(f"Loading model from {config.model_dir}")
    pipeline = DDPMPipeline.from_pretrained(config.model_dir)
    
    # --------------------float64修改地方-----------------------
    new_noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule="linear",prediction_type="epsilon",variance_type="fixed_large")

    # --------------------float64修改地方-----------------------

    # 将模型移动到 GPU，如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    # 设置模型为评估模式
    pipeline.unet.eval()

    # 随机采样噪声作为输入
    batch_size = config.infer_batch_size
    image_size = config.image_size
    num_images = config.num_images_to_generate
    num_inference_steps = 1000
    save_dir_id = config.save_dir_id
    
    
    # 设置调度器的时间步数
    pipeline.scheduler.set_timesteps(num_inference_steps)

    for n_batch in tqdm(range(0, num_images, batch_size), desc="generating"):
        # 确定当前批次的图像数量，最后一批次可能不足 batch_size
        current_batch_size = min(batch_size, num_images - n_batch)
        
        # 动态调整 image_shape 大小以适应当前批次
        image_shape = (current_batch_size, pipeline.unet.config.in_channels, config.image_size, config.image_size)
        images = torch.randn(image_shape, device=device)
        
        for i, t in tqdm(enumerate(pipeline.scheduler.timesteps), desc=f"generate batch {n_batch}"):
            with torch.no_grad():
                model_output = pipeline.unet(images, t).sample
                # images = pipeline.scheduler.step(model_output, t, images).prev_sample
                images = new_noise_scheduler.step(model_output, t, images).prev_sample
            
            # 在最后一个时间步保存图像
            if (i + 1) == num_inference_steps:
                # import ipdb;ipdb.set_trace();
                generated_images = images / 2 + 0.5  # 将值归一化到 [0, 1]
                generated_images = generated_images.clamp(0, 1).cpu()
                
                # 将生成的 Tensor 转为 PIL 图像
                for j in range(current_batch_size):  # 使用 current_batch_size 而不是 batch_size
                    image_tensor = generated_images[j]
                    image_pil = transforms.ToPILImage()(image_tensor)
                    # 生成文件名并保存图像
                    image_path = os.path.join(config.output_dir, f"img_{num_images*save_dir_id + n_batch + j + 1:05d}.png") # 这里编号可能需要修改 111
                    image_pil.save(image_path)



    print(f"Finished generating {num_images} images.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DDPM Image Generation")
    parser.add_argument("--save_dir_id", type=int, default=0, help="save directory id")
    
    # 解析命令行参数
    args = parser.parse_args()
    save_dir_id = args.save_dir_id

    class Config:
        model_dir =   "./ddpm-ema-cifar10-11-30-float64"  # 训练好的模型目录
        output_dir = f"./ddpm-ema-cifar10-11-30-float64/g_img_dir_{save_dir_id}" 
        infer_batch_size = 250  # 推理时的批次大小，一张张生成吧
        image_size = 32  # 图像尺寸
        num_images_to_generate = 6250  # 生成的图像数量
        save_dir_id = save_dir_id
    config = Config()
    evaluate(config)

# TODO： 上述方法和直接采用pipeline去比较一下效果