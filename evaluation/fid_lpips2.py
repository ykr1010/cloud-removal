import os
from glob import glob
import numpy as np
import lpips
import torch
import subprocess
import sys
from tqdm import tqdm
from torchvision.models import inception_v3
from torchvision import transforms
from scipy import linalg
import torch.nn.functional as F
from PIL import Image


def calculate_lpips(src, dst):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_list = []
    
    # 获取所有图片
    all_images = sorted(glob(os.path.join(src, "*")))
    
    if len(all_images) == 0:
        print("错误: 目录中没有找到图片!")
        return
    
    # 根据文件名前缀分类图片
    gt_images = sorted([img for img in all_images if os.path.basename(img).startswith('GT')])
    out_images = sorted([img for img in all_images if os.path.basename(img).startswith('Out')])
    
    print(f"找到 {len(gt_images)} 张原始图片和 {len(out_images)} 张生成图片")
    
    if len(gt_images) == 0 or len(out_images) == 0:
        print("错误: 没有找到匹配的图片对!")
        print("请确保图片文件名以'GT'和'Out'开头")
        return
    
    if len(gt_images) != len(out_images):
        print(f"错误: 图片数量不匹配!\n原始图片数量: {len(gt_images)}\n生成图片数量: {len(out_images)}")
        return
    
    print("开始计算LPIPS...")
    for i, (gt_path, out_path) in enumerate(tqdm(zip(gt_images, out_images), total=len(gt_images))):
        try:
            img1 = lpips.im2tensor(lpips.load_image(gt_path)).to(device)
            img2 = lpips.im2tensor(lpips.load_image(out_path)).to(device)
            
            if img1.shape != img2.shape:
                print(f"警告: 图片 {i+1} 尺寸不匹配")
                print(f"GT图片: {gt_path}, 尺寸: {img1.shape}")
                print(f"Out图片: {out_path}, 尺寸: {img2.shape}")
                continue
                
            lpips_value = loss_fn_alex(img1, img2).item()
            lpips_list.append(lpips_value)
            
        except Exception as e:
            print(f"处理图片 {i+1} 时出错: {str(e)}")
            continue
    
    if len(lpips_list) > 0:
        print(f"\nLPIPS结果: {np.mean(lpips_list):.4f}")
    else:
        print("错误: 没有成功处理任何图片!")


def get_inception_features(images, model, device):
    features = []
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    for img_path in tqdm(images, desc="提取特征"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(img)
            features.append(feature.cpu().numpy())
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")
            continue
    
    return np.concatenate(features, axis=0)


def calculate_fid(src, dst):
    try:
        print("开始计算FID...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载Inception模型
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()  # 移除最后的全连接层
        model = model.to(device)
        model.eval()
        
        # 获取所有图片
        all_images = sorted(glob(os.path.join(src, "*")))
        gt_images = sorted([img for img in all_images if os.path.basename(img).startswith('GT')])
        out_images = sorted([img for img in all_images if os.path.basename(img).startswith('Out')])
        
        # 提取特征
        gt_features = get_inception_features(gt_images, model, device)
        out_features = get_inception_features(out_images, model, device)
        
        # 计算均值和协方差
        mu_gt = np.mean(gt_features, axis=0)
        mu_out = np.mean(out_features, axis=0)
        sigma_gt = np.cov(gt_features, rowvar=False)
        sigma_out = np.cov(out_features, rowvar=False)
        
        # 计算FID
        ssdiff = np.sum((mu_gt - mu_out) ** 2.0)
        covmean = linalg.sqrtm(sigma_gt.dot(sigma_out))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma_gt + sigma_out - 2.0 * covmean)
        
        print(f"\nFID结果: {fid:.4f}")
        
    except Exception as e:
        print(f"计算FID时出错: {str(e)}")


if __name__ == "__main__":
    # 设置图片目录
    img_dir = "E:/administrator/DiffCR-main/experiments/test_nafnet_double_encoder_splitcaCond_splitcaUnet_250627_031246/results/test/0"
    
    # 确保目录存在
    if not os.path.exists(img_dir):
        print(f"错误: 目录不存在! {img_dir}")
        print("请确保您在正确的目录下运行此脚本")
        print("当前工作目录:", os.getcwd())
        exit(1)
    
    # 计算LPIPS
    calculate_lpips(img_dir, img_dir)
    
    # 计算FID
    calculate_fid(img_dir, img_dir)
