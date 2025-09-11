import h5py
import numpy as np
import cv2

def create_side_by_side_video(file_path, output_video_path, agent_key="data/demo_0/obs/agentview_rgb", eye_key="data/demo_0/obs/eye_in_hand_rgb"):
    """
    从HDF5文件中提取agentview_rgb和eye_in_hand_rgbd，生成左右拼接的视频并保存
    :param file_path: HDF5文件路径
    :param output_video_path: 输出视频路径（.mp4）
    :param agent_key: agentview_rgb数据集的key
    :param eye_key: eye_in_hand_rgbd数据集的key
    """
    try:
        # ✅ 打开HDF5文件
        with h5py.File(file_path, 'r') as f:
            # ✅ 检查数据集是否存在
            if agent_key not in f or eye_key not in f:
                print(f"❌ 数据集 '{agent_key}' 或 '{eye_key}' 不存在！")
                return
            
            # ✅ 读取数据集
            agent_data = f[agent_key][()]
            eye_data = f[eye_key][()]
            
            # ✅ 检查数据形状
            if agent_data.shape[0] != eye_data.shape[0]:
                print(f"❌ 两数据集帧数不一致！agentview_rgb: {agent_data.shape[0]}, eye_in_hand_rgbd: {eye_data.shape[0]}")
                return
            
            # ✅ 处理RGB和RGBD数据
            if len(agent_data.shape) != 4 or agent_data.shape[3] != 3:
                print(f"❌ agentview_rgb 数据形状异常: {agent_data.shape}，期望 (T, H, W, 3)")
                return
            
            # 如果eye_in_hand_rgbd是RGBD（4通道），提取RGB部分
            if len(eye_data.shape) == 4 and eye_data.shape[3] == 4:
                print("📌 检测到eye_in_hand_rgbd为RGBD，提取RGB通道")
                eye_data = eye_data[:, :, :, :3]  # 取前3个通道（RGB）
            elif len(eye_data.shape) != 4 or eye_data.shape[3] != 3:
                print(f"❌ eye_in_hand_rgbd 数据形状异常: {eye_data.shape}，期望 (T, H, W, 3) 或 (T, H, W, 4)")
                return
            
            # ✅ 确保数据类型为uint8
            if agent_data.dtype != np.uint8:
                agent_data = agent_data.astype(np.uint8)
            if eye_data.dtype != np.uint8:
                eye_data = eye_data.astype(np.uint8)
            
            # ✅ 获取视频参数
            num_frames, height, width, _ = agent_data.shape
            fps = 30  # 可根据需要调整帧率
            output_size = (width * 2, height)  # 左右拼接，宽度翻倍
            
            # ✅ 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
            out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)
            
            # ✅ 逐帧拼接并写入视频
            for i in range(num_frames):
                # 左右拼接图像
                combined_frame = np.hstack((agent_data[i], eye_data[i]))
                # OpenCV使用BGR格式，需转换RGB到BGR
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                out.write(combined_frame)
                print(f"📽️ 已处理帧 {i+1}/{num_frames}")
            
            # ✅ 释放视频写入器
            out.release()
            print(f"🎉 视频已保存到: {output_video_path}")
            
    except FileNotFoundError:
        print("❌ 文件不存在，请检查文件路径！")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")

# 示例用法
if __name__ == "__main__":
    file_path = "/home/sythoid_01/文档/Huangwenlong/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    output_video_path = "/home/sythoid_01/文档/Huangwenlong/LIBERO/custom/output_video.mp4"  # 输出视频文件名
    create_side_by_side_video(file_path, output_video_path)