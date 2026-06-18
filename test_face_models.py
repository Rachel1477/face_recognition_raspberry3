"""
人脸检测和识别模型测试脚本
测试 YuNet 检测器 + SFace 识别器是否正常工作
"""

import cv2
import numpy as np
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.face_utils import face_processor


def test_face_detection(image_path=None, save_result=True):
    """
    测试人脸检测功能
    
    Args:
        image_path: 图片路径，如果为None则使用摄像头
        save_result: 是否保存结果图片
    """
    print("=" * 60)
    print("人脸检测和识别模型测试")
    print("=" * 60)
    
    # 检查模型是否加载成功
    if face_processor.detector is None:
        print("❌ 检测器未加载!")
        return False
    else:
        print("✓ 检测器加载成功")
    
    if face_processor.recognizer is None:
        print("❌ 识别器未加载!")
        return False
    else:
        print("✓ 识别器加载成功")
    
    print("-" * 60)
    
    # 读取图片或打开摄像头
    if image_path and os.path.exists(image_path):
        print(f"测试图片: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ 无法读取图片: {image_path}")
            return False
    else:
        print("使用摄像头测试...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print("❌ 无法从摄像头获取图像")
            return False
    
    print(f"图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
    print("-" * 60)
    
    # 设置检测器输入尺寸
    height, width = frame.shape[:2]
    face_processor.detector.setInputSize((width, height))
    
    # 检测人脸
    print("正在检测人脸...")
    faces = face_processor.detector.detect(frame)
    
    if faces[1] is None or len(faces[1]) == 0:
        print("❌ 未检测到人脸!")
        print("\n可能的原因:")
        print("  1. 图片中没有人脸")
        print("  2. 人脸太小或太模糊")
        print("  3. 光线太暗")
        print("  4. 模型文件损坏")
        
        # 保存原图用于调试
        if save_result:
            result_path = "test_result_no_face.jpg"
            cv2.imwrite(result_path, frame)
            print(f"\n已保存原图到: {result_path}")
        return False
    
    # 获取检测到的人脸
    detected_faces = faces[1]
    print(f"✓ 检测到 {len(detected_faces)} 张人脸")
    
    # 复制图像用于绘制
    result_frame = frame.copy()
    
    # 处理每张人脸
    for i, face in enumerate(detected_faces):
        print(f"\n--- 人脸 {i + 1} ---")
        
        # 获取人脸边界框
        box = face[:4].astype(int)
        x, y, w, h = box
        print(f"位置: ({x}, {y}, {w}, {h})")
        
        # 获取关键点
        landmarks = face[4:14].reshape(5, 2).astype(int)
        print(f"关键点: {landmarks.tolist()}")
        
        # 绘制人脸边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
        
        # 绘制关键点
        for j, (lx, ly) in enumerate(landmarks):
            cv2.circle(result_frame, (lx, ly), 3, (0, 0, 255), -1)  # 红色点
        
        # 绘制人脸编号
        cv2.putText(result_frame, f"Face {i + 1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 对齐人脸
        aligned_face = face_processor.recognizer.alignCrop(frame, face)
        
        # 提取特征
        feature = face_processor.recognizer.feature(aligned_face)
        
        if feature is not None:
            print(f"特征向量维度: {feature.shape}")
            print(f"特征向量范数: {np.linalg.norm(feature):.4f}")
            
            # 绘制特征提取成功标记
            cv2.putText(result_frame, "OK", (x + w - 30, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print("❌ 特征提取失败")
            cv2.putText(result_frame, "FAIL", (x + w - 50, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 保存结果
    if save_result:
        result_path = "test_result.jpg"
        cv2.imwrite(result_path, result_frame)
        print(f"\n✓ 结果已保存到: {result_path}")
        print("  - 绿色框: 检测到的人脸")
        print("  - 红色点: 人脸关键点")
        print("  - OK: 特征提取成功")
    
    # 显示结果
    print("\n按任意键关闭预览窗口...")
    cv2.imshow("Face Detection Test", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def test_feature_matching():
    """
    测试特征匹配功能
    比较两张人脸图片的相似度
    """
    print("\n" + "=" * 60)
    print("人脸特征匹配测试")
    print("=" * 60)
    
    # 测试图片目录
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        print(f"请创建 {test_dir} 目录并放入测试图片")
        return
    
    # 获取所有图片
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < 2:
        print(f"请在 {test_dir} 目录中放入至少 2 张图片")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 提取所有特征
    features = []
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ 无法读取: {img_file}")
            continue
        
        feature = face_processor.extract_face_features(img)
        if feature is not None:
            features.append((img_file, feature))
            print(f"✓ {img_file}: 特征提取成功")
        else:
            print(f"❌ {img_file}: 未检测到人脸")
    
    if len(features) < 2:
        print("需要至少 2 张成功提取特征的人脸图片")
        return
    
    # 计算相似度矩阵
    print("\n相似度矩阵:")
    print("-" * 60)
    
    # 打印表头
    header = " " * 20
    for name, _ in features:
        header += f"{name[:15]:>18}"
    print(header)
    
    # 计算并打印相似度
    for i, (name1, feat1) in enumerate(features):
        row = f"{name1[:18]:<18}"
        for j, (name2, feat2) in enumerate(features):
            similarity = face_processor.cosine_similarity(feat1, feat2)
            row += f"{similarity:>18.4f}"
        print(row)
    
    print("-" * 60)
    print("相似度 > 0.36 通常表示同一人")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="人脸检测和识别模型测试")
    parser.add_argument("-i", "--image", help="测试图片路径")
    parser.add_argument("-m", "--matching", action="store_true", 
                       help="测试特征匹配")
    parser.add_argument("--no-save", action="store_true",
                       help="不保存结果图片")
    
    args = parser.parse_args()
    
    if args.matching:
        test_feature_matching()
    else:
        test_face_detection(args.image, not args.no_save)


if __name__ == "__main__":
    main()
