# text-to-image-debiasing

文生图去偏-元学习

meta_unet.py是元学习版本的UNet2DConditionModel，继承MetaModule

train_sdxl_with_metaunet.py是sdxl训练代码，针对不同属性的去偏只需设计好少量元数据集，然后修改此代码中的数据加载函数即可

test_meta_sdxl.py是使用训练好的模型进行图片生成

⚠️注意，训练好的模型下面的unet/config.json需要手动替换成附带的config.json

进行训练执行下面指令即可


CUDA_VISIBLE_DEVICES=9 python train_sdxl_with_metaunet.py   --pretrained_model_name_or_path ./sdxl   --use_ema   --resolution=64   --center_crop   --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=1000   --learning_rate=1e-05   --max_grad_norm=1   --seed=42   --lr_scheduler="constant"   --lr_warmup_steps=0   --output_dir="sd-naruto-model_metahh"


my_mwnet.py是我对Meta weight net文章的复现代码
