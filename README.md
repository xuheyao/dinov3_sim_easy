# dinov3_sim_easy
dinov3 sim by one easy script for beginers
![ex1](https://github.com/xuheyao/dinov3_sim_easy/blob/main/%E6%88%AA%E5%9B%BE%202025-12-16%2010-37-38.png)
![ex2](https://github.com/xuheyao/dinov3_sim_easy/blob/main/%E6%88%AA%E5%9B%BE%202025-12-16%2010-38-02.png)
## 设置全局变量
### 添加DINOV3_LOCATION
1. git clone https://github.com/facebookresearch/dinov3
2. 在系统环境中添加 DINOV3_LOCATION = dinov3_path
修改
- DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", "/home/macoofi/dinov3/dinov3")
### 添加模型地址,dinov3是一个万金油工具,我放在了 "DINOV3_LOCATION/backbone" 里面,你可以任意选择
修改
- CHECKPOINT_PATH = f"{DINOV3_LOCATION}/backbone/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
### 修改模型类型,不同的vit有不同的层数,就是模型文件名第二个"_"的内容
修改
- MODEL_NAME = "dinov3_vitb16"

### 直接跳到文件最底部
你已经可以运行了

加载图像
- img, orig_bgr = load_image("image.png")
你需要自己传入图片

调整图像大小
- img_resized = resize_image(img,320)
vit模型以16*16为一个为一个block,这里有32/16^2 = 20*20个block,足够你的选择了
返回的是400*768的features,转换为 768个20*20特征图

