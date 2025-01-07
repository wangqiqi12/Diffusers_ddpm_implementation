### diffusers库复现ddpm在cifar10的FID踩坑总结

注：基于https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py的代码修改

复现仓库：https://github.com/wangqiqi12/Diffusers_ddpm_implementation/tree/main



1.Unet模型架构，针对不同数据集需要采用不同的网络层数

具体体现在UNet2DModel的设置上：

```python
    if args.resolution == 32: # 比如针对cifar10数据集，Unet架构如下
        model = UNet2DModel(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            dropout=0.1,
            block_out_channels=(128, 256, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )
```



2. EMA：EMA对Unet模型推理的效果影响比较大，需要使用EMAmodel来平滑模型参数，同时注意EMA更新方式、所有参数设置需要与原论文代码一致

```python
# Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=False, # 这里也是坑点，不能设置为True(否则ema更新公式会发生改变)，可以查看diffusers库里EMAModel的代码
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
```



3. batchsize：注意多卡训练的batchsize设置。代码的batchsize设置方式是设置总batchsize，还是每个设备上的batchsize都需要弄清楚。该份代码采用的是每个device上的batchsize。同时还需注意梯度累积的step也会影响。

```python
parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

……

total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
# 这个才是计算总batchsize的正确方式
```



4. 精度：fp16对于cifar10是不够的，一般不用mixed_precision=fp16;float32和float64有细微区别，在该份代码中特意和原论文代码对照将部分地方换成float64。

```python
if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float64)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
```



5. 数据集的格式转换：dataset的transform.compose里的变换需要与原论文代码一致，尤其是归一化的细节（比如cifar10在图像生成领域的归一化是[0.5,0.5,0.5]，而在图像分类的归一化则是另外三个四位小数）

```python
augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3), # 注意这个归一化
        ]
    )
    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}

    dataset.set_transform(transform_images)
```



6. 随机种子：多卡采样时如果没有写好accelerate多卡采样，在多卡跑相同代码时不要设置在代码里设置固定的random_seed

7. 学习率：learning_rate不仅仅只关注设置的值，还要关注学习率变化的设置(具体对应lr_scheduler，查看学习率是constant还是warmup等等)

```python
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument( 
    "--lr_scheduler",
    type=str,
    default="cosine", # 原diffusers代码的坑点！
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
)
```

8. 优化器：optimizer的所有参数值设置是否一致，微小的差别是否会带来比较大的影响

```python
parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.") # 注意修改！与原论文代码一致！
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0, help="Weight decay magnitude for the Adam optimizer." 
    )# 注意这个修改！与原论文代码一致！
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
```





最终执行复现成功的仓库训练代码的命令如下：

```bash
accelerate launch --multi_gpu train_unconditional_float64.py \
  --dataset_name="uoft-cs/cifar10" \
  --resolution=32 --center_crop --random_flip \
  --output_dir="ddpm-ema-cifar10-float64" \
  --train_batch_size=16 \
  --num_epochs=2048 \
  --save_images_epochs=500 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=2e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=5000 \
  --logger="tensorboard" \
  --checkpointing_steps=200000
```



