from flask import Flask, request, jsonify, send_from_directory
import os
import h5py
import numpy as np
import base64
import tempfile
import subprocess
import shutil
import cv2
import re

app = Flask(__name__, static_folder='.')
BASE_PATHS = {
    'primitive': '/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vlabench/pretrain/primitive',
    'composite': '/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vlabench/pretrain/composite',
}

def list_tasks(task_type):
    root = BASE_PATHS[task_type]
    tasks = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(tasks)

def list_hdf5_files(task_type, task):
    task_dir = os.path.join(BASE_PATHS[task_type], task)
    hdf5_files = [f for f in os.listdir(task_dir) if f.endswith('.hdf5')]
    return sorted(hdf5_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

def load_instruction_and_video(filepath, n_frame=20, fps=5):
    try:
        with h5py.File(filepath, 'r') as f:
            data = f["data"]
            timestamps = list(data.keys())
            if not timestamps:
                return {"error": "No timestamps in file"}
            ts = timestamps[0]
            
            # 读取instruction（保持不变）
            instruction_raw = data[ts]["instruction"]
            
            if isinstance(instruction_raw, h5py.Dataset):
                instruction_array = np.array(instruction_raw)
                if instruction_array.dtype.kind in ['S', 'U']:  # 字符串类型
                    if instruction_array.ndim == 0:  # 标量
                        instruction = str(instruction_array.item())
                        if isinstance(instruction_array.item(), bytes):
                            instruction = instruction_array.item().decode('utf-8')
                    else:  # 数组
                        instruction = []
                        for item in instruction_array.flat:
                            if isinstance(item, bytes):
                                instruction.append(item.decode('utf-8'))
                            else:
                                instruction.append(str(item))
                else:
                    instruction = instruction_array.tolist()
            else:
                instruction = str(instruction_raw)
            
            rgbs = np.array(data[ts]["observation"]["rgb"])  # [T, N, C, H, W], uint8
            print(f"RGB shape: {rgbs.shape}, dtype: {rgbs.dtype}")
            print(f"RGB value range: min={rgbs.min()}, max={rgbs.max()}")
            
            n_frame = min(n_frame, rgbs.shape[0])
            n_views = rgbs.shape[1]
            
            # 检查FFmpeg是否可用
            if not shutil.which('ffmpeg'):
                return {"error": "FFmpeg not found. Please install FFmpeg."}
            
            # 为每个视角生成MP4视频
            videos_data = []
            
            for view_idx in range(n_views):
                try:
                    print(f"Processing view {view_idx} with FFmpeg")
                    
                    # 创建临时目录保存帧
                    temp_dir = tempfile.mkdtemp()
                    frame_files = []
                    
                    # 准备并保存所有帧为PNG文件
                    for frame_idx in range(n_frame):
                        img = rgbs[frame_idx, view_idx]  # [C, H, W]
                        
                        # 转换为 [H, W, C] 格式
                        if len(img.shape) == 3 and img.shape[0] == 3:
                            img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
                        
                        # 处理数值范围
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            if img.max() <= 1.0 and img.min() >= 0.0:
                                img = (img * 255.0).astype(np.uint8)
                            elif img.max() <= 255.0 and img.min() >= 0.0:
                                img = img.astype(np.uint8)
                            else:
                                img = img - img.min()
                                img = img / img.max() if img.max() > 0 else img
                                img = (img * 255.0).astype(np.uint8)
                        elif img.dtype != np.uint8:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                        
                        # 确保是3通道RGB
                        if len(img.shape) == 2:
                            img = np.stack([img, img, img], axis=2)
                        elif len(img.shape) == 3 and img.shape[2] == 1:
                            img = np.repeat(img, 3, axis=2)
                        elif len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        
                        # 保存为PNG文件
                        frame_filename = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
                        cv2.imwrite(frame_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        frame_files.append(frame_filename)
                        
                        if frame_idx == 0:
                            print(f"First frame: shape={img.shape}, dtype={img.dtype}, range={img.min()}-{img.max()}")
                    
                    print(f"Saved {len(frame_files)} frames to {temp_dir}")
                    
                    # 创建输出视频文件
                    output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    output_filename = output_video.name
                    output_video.close()
                    
                    # 使用FFmpeg创建视频
                    ffmpeg_cmd = [
                        'ffmpeg', '-y',  # -y 覆盖输出文件
                        '-framerate', str(fps),  # 输入帧率
                        '-i', os.path.join(temp_dir, 'frame_%04d.png'),  # 输入文件模式
                        '-c:v', 'libx264',  # 使用H.264编码器
                        '-pix_fmt', 'yuv420p',  # 像素格式，兼容性最好
                        '-crf', '23',  # 质量控制，23是好的平衡点
                        '-preset', 'medium',  # 编码预设
                        '-movflags', '+faststart',  # 优化网络播放
                        '-r', str(fps),  # 输出帧率
                        output_filename
                    ]
                    
                    print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                    
                    # 执行FFmpeg命令
                    result = subprocess.run(
                        ffmpeg_cmd, 
                        capture_output=True, 
                        text=True,
                        timeout=60  # 60秒超时
                    )
                    
                    if result.returncode == 0:
                        print(f"FFmpeg succeeded for view {view_idx}")
                        
                        # 检查输出文件
                        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 1000:
                            file_size = os.path.getsize(output_filename)
                            print(f"Video file created: {file_size} bytes")
                            
                            # 验证视频
                            cap = cv2.VideoCapture(output_filename)
                            if cap.isOpened():
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                video_fps = cap.get(cv2.CAP_PROP_FPS)
                                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                cap.release()
                                
                                print(f"Video info: {frame_count} frames, {video_fps} fps, {video_width}x{video_height}")
                                
                                if frame_count > 0:      
                                     # 读取视频文件并编码为base64
                                    with open(output_filename, 'rb') as video_file:
                                        video_data = video_file.read()
                                        video_b64 = base64.b64encode(video_data).decode('utf-8')
                                    
                                    videos_data.append({
                                        'view_id': view_idx,
                                        'video_b64': video_b64,
                                        'frames_count': n_frame,
                                        'codec': 'h264',
                                        'video_info': {
                                            'width': video_width,
                                            'height': video_height,
                                            'fps': video_fps,
                                            'frame_count': frame_count,
                                            'file_size': file_size
                                        }
                                    })
                                    
                                    print(f"Successfully processed view {view_idx} with FFmpeg")
                                else:
                                    print(f"Video created but no frames readable for view {view_idx}")
                            else:
                                print(f"Cannot open created video for view {view_idx}")
                        else:
                            print(f"Video file not created or too small for view {view_idx}")
                    else:
                        print(f"FFmpeg failed for view {view_idx}")
                        print(f"FFmpeg stderr: {result.stderr}")
                        print(f"FFmpeg stdout: {result.stdout}")
                    
                    # 清理临时文件
                    try:
                        shutil.rmtree(temp_dir)
                        if os.path.exists(output_filename):
                            os.unlink(output_filename)
                    except Exception as cleanup_error:
                        print(f"Cleanup error: {cleanup_error}")
                    
                except subprocess.TimeoutExpired:
                    print(f"FFmpeg timeout for view {view_idx}")
                    continue
                except Exception as e:
                    print(f"Error creating video for view {view_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"Successfully created {len(videos_data)} videos out of {n_views} views using FFmpeg")
            
            return {
                "instruction": instruction,
                "total_frames": int(rgbs.shape[0]),
                "n_views": int(n_views),
                "videos_data": videos_data,
                "fps": fps,
                "encoder": "ffmpeg"
            }
    except Exception as e:
        print(f"General error in load_instruction_and_video: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
              
def count_hdf5_files(root_dir):
    result = {}
    for subdir in os.listdir(root_dir):
        full_path = os.path.join(root_dir, subdir)
        if os.path.isdir(full_path):
            count = sum(f.endswith('.hdf5') for f in os.listdir(full_path))
            result[subdir] = count
    return result

@app.route('/tasks')
def get_tasks():
    task_type = request.args.get('type')
    if task_type not in BASE_PATHS:
        return jsonify([])  # invalid
    return jsonify(list_tasks(task_type))

@app.route('/files')
def get_files():
    task_type = request.args.get('type')
    task = request.args.get('task')
    if task_type not in BASE_PATHS or not task:
        return jsonify([])
    return jsonify(list_hdf5_files(task_type, task))

@app.route('/hdf5_content')
def get_hdf5_content():
    task_type = request.args.get('type')
    task = request.args.get('task')
    filename = request.args.get('file')
    n_frame = int(request.args.get('n_frame', 20))
    fps = int(request.args.get('fps', 5))
    
    if task_type not in BASE_PATHS or not task or not filename:
        return jsonify({'error': 'bad request'})
    
    filepath = os.path.join(BASE_PATHS[task_type], task, filename)
    result = load_instruction_and_video(filepath, n_frame, fps)
    
    return jsonify(result)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/stats')
def get_stats():
    task_type = request.args.get('type')
    if task_type not in BASE_PATHS:
        return jsonify({'error': 'invalid type'}), 400
    stats = count_hdf5_files(BASE_PATHS[task_type])
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8500, debug=True)