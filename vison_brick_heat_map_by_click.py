import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# 全局变量
DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", "/home/macoofi/dinov3/dinov3")
CHECKPOINT_PATH = f"{DINOV3_LOCATION}/backbone/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
MODEL_NAME = "dinov3_vitb16"

# 全局状态变量
model = None
features = None
grid_img = None
actual_patch_size = None
grid_size = None
RESIZE = 640
ax1 = None
ax2 = None


def load_model():
    """加载DINOv3模型"""
    global model
    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local",
        weights=CHECKPOINT_PATH,
    )
    model.eval().cuda()
    print("Model loaded successfully.")
    return model


def load_image(img_path="190223 mayl3.png"):
    """加载图像"""
    orig_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image loaded from {img_path}")
    return img, orig_bgr


def resize_image(img, resize_size=640):
    """调整图像大小为指定的尺寸"""
    global RESIZE, actual_patch_size, grid_size
    RESIZE = resize_size
    print(f"Use {RESIZE}x{RESIZE} which is exactly {RESIZE//16}x{RESIZE//16} patches of 16x16")
    img_resized = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
    PATCH_SIZE = RESIZE // 16
    return img_resized


def get_features(img_resized):
    """从图像中提取patch特征"""
    global features, grid_size, actual_patch_size
    
    print("Preprocess for DINOv3")
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 添加batch维度并应用变换
    inp = transform(img_resized).unsqueeze(0).cuda()
    print(f"Input shape: {inp.shape}")
    
    print("Extract patch embeddings (features)")
    with torch.no_grad():
        features_dict = model.forward_features(inp)
        features = features_dict['x_norm_patchtokens'][0].cpu().numpy()
        print("size of the features:", features.shape)
    
    # 检查实际的patch数量
    num_patches = features.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    # 计算显示时的实际patch大小
    actual_patch_size = RESIZE // grid_size
    
    return features


def render_grid(img_resized):
    """在图像上绘制patch网格"""
    global grid_img
    
    grid_img = img_resized.copy()
    for i in range(1, grid_size):
        x = i * actual_patch_size
        cv2.line(grid_img, (x, 0), (x, RESIZE), (255, 0, 0), 2)
    for j in range(1, grid_size):
        y = j * actual_patch_size
        cv2.line(grid_img, (0, y), (RESIZE, y), (255, 0, 0), 2)
    
    return grid_img


def generate_similarity_map(patch_idx):
    """为给定的patch索引生成相似度图"""
    # 获取参考特征
    reference_feature = features[patch_idx]
    
    """
    计算与所有patch的余弦相似度
    用点积除以范数的乘积来归一化
    @ --> 点积
    np.linalg.norm --> l2范数
    """
    similarities = features @ reference_feature / (np.linalg.norm(features, axis=1) * np.linalg.norm(reference_feature) + 1e-8)
    similarities = similarities.reshape(grid_size, grid_size)
    
    # 调整相似度图大小以匹配图像大小
    sim_resized = cv2.resize(similarities, (RESIZE, RESIZE), interpolation=cv2.INTER_CUBIC)
    sim_norm = cv2.normalize(sim_resized, None, 0, 255, cv2.NORM_MINMAX)
    sim_color = cv2.applyColorMap(sim_norm.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    
    # 计算patch坐标
    patch_y = patch_idx // grid_size
    patch_x = patch_idx % grid_size
    
    # 用矩形标记点击的patch
    marked_img = sim_color.copy()
    top_left = (patch_x * actual_patch_size, patch_y * actual_patch_size)
    bottom_right = ((patch_x + 1) * actual_patch_size, (patch_y + 1) * actual_patch_size)
    
    # 在目标patch上画点
    center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
    cv2.circle(marked_img, center, radius=5, color=(0, 0, 255), thickness=-1)
    
    return marked_img, (patch_y, patch_x)


def on_click(event):
    """处理图像上的鼠标点击事件"""
    global ax1, ax2
    
    if event.inaxes != ax1:
        return
    
    # 获取点击坐标
    x, y = int(event.xdata), int(event.ydata)
    
    # 计算点击了哪个patch
    patch_x = min(x // actual_patch_size, grid_size - 1)
    patch_y = min(y // actual_patch_size, grid_size - 1)
    idx = patch_y * grid_size + patch_x
    
    print(f"Clicked patch ({patch_y}, {patch_x}), index: {idx}")
    print(f"Features shape: {features.shape}")
    print(f"Reference feature shape: {features[idx].shape}")
    
    # 生成相似度图
    sim_img, coords = generate_similarity_map(idx)
    
    # 更新相似度图显示
    ax2.clear()
    ax2.imshow(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Cosine Similarity Map - Patch {coords}')
    ax2.axis('off')
    
    # 在原图上画矩形显示选中的patch
    ax1.clear()
    ax1.imshow(grid_img)
    rect = plt.Rectangle((patch_x * actual_patch_size, patch_y * actual_patch_size), 
                        actual_patch_size, actual_patch_size, 
                        fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)
    ax1.set_title('DINOv3 Patches (click to select)')
    ax1.axis('off')
    
    plt.draw()


def show():
    """显示交互式界面"""
    global ax1, ax2
    
    # 创建带有两个子图的图形
    fig, (ax1_local, ax2_local) = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = ax1_local, ax2_local
    
    # 显示网格图像
    ax1.imshow(grid_img)
    ax1.set_title('DINOv3 Patches (click to select)')
    ax1.axis('off')
    
    # 初始化相似度图显示
    ax2.imshow(grid_img)
    ax2.set_title('Cosine Similarity Map')
    ax2.axis('off')
    
    # 连接点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数 - 按顺序执行所有步骤"""
    # 1. 加载模型
    load_model()
    
    # 2. 加载图像
    img, orig_bgr = load_image("image.png")
    
    # 3. 调整图像大小
    img_resized = resize_image(img,320)
    
    # 4. 提取特征
    get_features(img_resized)
    
    # 5. 渲染网格
    render_grid(img_resized)
    
    # 6. 显示交互式界面
    show()


if __name__ == "__main__":
    main()
