# ---
文生图去偏-元学习
meta_unet.py是元学习版本的UnetConditional2d，继承MetaModule
train_sdxl_with_metaunet.py是sdxl训练代码，针对不同属性的去偏只需设计好少量元数据集，然后修改此代码中的数据加载函数即可
test_meta_sdxl.py是使用训练好的模型进行图片生成
⚠️注意，训练好的模型下面的unet/config.json需要手动替换成附带的config.json

my_mwnet.py是我对Meta weight net文章的复现代码
