# import numpy as np
# import json
# import time
# from scipy.optimize import minimize
# from scipy.spatial.distance import euclidean

# class PoseEstimator3D:
#     def __init__(self):
#         self.coco_skeleton = [
#             [0, 1], [0, 2], [1, 3], [2, 4],
#             [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
#             [5, 11], [6, 12], [11, 12],
#             [11, 13], [13, 15], [12, 14], [14, 16]
#         ]
        
#         self.avg_bone_lengths = {
#             (0, 1): 0.08, (0, 2): 0.08, (1, 3): 0.12, (2, 4): 0.12,
#             (5, 6): 0.38, (5, 7): 0.28, (7, 9): 0.28, (6, 8): 0.28, (8, 10): 0.28,
#             (5, 11): 0.35, (6, 12): 0.35, (11, 12): 0.24,
#             (11, 13): 0.42, (13, 15): 0.43, (12, 14): 0.42, (14, 16): 0.43
#         }
        
#     def estimate_camera_params(self, keypoints_2d, ankle_world, ankle_indices=[15, 16]):
#         valid_kpts = keypoints_2d[keypoints_2d[:, 2] > 0.3]
#         if len(valid_kpts) == 0:
#             return None, None, 1920/2, 1080/2
        
#         center_2d = np.mean(valid_kpts[:, :2], axis=0)
#         scale_2d = np.std(valid_kpts[:, :2])
        
#         ankle_2d = keypoints_2d[ankle_indices]
#         ankle_world_pos = np.array(ankle_world)
        
#         if np.all(ankle_2d[:, 2] > 0.3):
#             ankle_center_2d = np.mean(ankle_2d[:, :2], axis=0)
#             ankle_dist_2d = np.linalg.norm(ankle_2d[0, :2] - ankle_2d[1, :2])
#             ankle_dist_3d = np.linalg.norm(ankle_world_pos[0] - ankle_world_pos[1])
            
#             if ankle_dist_2d > 0 and ankle_dist_3d > 0:
#                 focal_length = 1920
#                 scale = focal_length * ankle_dist_3d / (ankle_dist_2d + 1e-6)
#                 return focal_length, scale, center_2d[0], center_2d[1]
        
#         focal_length = 1920
#         scale = 2.0
#         return focal_length, scale, center_2d[0], center_2d[1]
    
#     def initialize_3d_pose(self, keypoints_2d, ankle_world, focal_length, scale, cx, cy):
#         pose_3d = np.zeros((17, 3))
        
#         left_ankle_idx, right_ankle_idx = 15, 16
#         pose_3d[left_ankle_idx] = [ankle_world[0][0], ankle_world[0][1], 0.04]
#         pose_3d[right_ankle_idx] = [ankle_world[1][0], ankle_world[1][1], 0.04]
        
#         pelvis_2d = (keypoints_2d[11, :2] + keypoints_2d[12, :2]) / 2
#         pelvis_x = ((pelvis_2d[0] - cx) * scale) / focal_length
#         pelvis_y = ((pelvis_2d[1] - cy) * scale) / focal_length
#         ankle_center = (pose_3d[left_ankle_idx, :2] + pose_3d[right_ankle_idx, :2]) / 2
        
#         pelvis_world = ankle_center + np.array([pelvis_x * 0.1, pelvis_y * 0.1])
#         pose_3d[11] = [pelvis_world[0], pelvis_world[1], 0.90]
#         pose_3d[12] = [pelvis_world[0], pelvis_world[1], 0.90]
        
#         for i in range(17):
#             if i not in [11, 12, 15, 16] and keypoints_2d[i, 2] > 0.3:
#                 depth_ratio = 1.0 - (keypoints_2d[i, 1] / 1080.0)
                
#                 x_norm = (keypoints_2d[i, 0] - cx) / focal_length
#                 y_norm = (keypoints_2d[i, 1] - cy) / focal_length
                
#                 if i <= 4:
#                     z = 1.5 + depth_ratio * 0.2
#                 elif i <= 10:
#                     z = 1.2 + depth_ratio * 0.3
#                 else:
#                     z = 0.5 + depth_ratio * 0.5
                
#                 pose_3d[i, 0] = pelvis_world[0] + x_norm * scale * z / 2.0
#                 pose_3d[i, 1] = pelvis_world[1] + y_norm * scale * z / 2.0
#                 pose_3d[i, 2] = z
        
#         return pose_3d
    
#     def refine_3d_pose(self, pose_3d_init, keypoints_2d, ankle_world, focal_length, scale, cx, cy):
#         def project_3d_to_2d(pose_3d, f, cx, cy):
#             projected = np.zeros((17, 2))
#             for i in range(17):
#                 if pose_3d[i, 2] > 0.01:
#                     projected[i, 0] = (f * pose_3d[i, 0] / pose_3d[i, 2]) + cx
#                     projected[i, 1] = (f * pose_3d[i, 1] / pose_3d[i, 2]) + cy
#             return projected
        
#         def loss_function(params):
#             pose_3d = params.reshape(17, 3)
            
#             reprojection_loss = 0
#             count = 0
#             projected = project_3d_to_2d(pose_3d, focal_length, cx, cy)
#             for i in range(17):
#                 if keypoints_2d[i, 2] > 0.3:
#                     weight = keypoints_2d[i, 2]
#                     diff = projected[i] - keypoints_2d[i, :2]
#                     reprojection_loss += weight * np.sum(diff ** 2)
#                     count += 1
#             if count > 0:
#                 reprojection_loss /= (count * 1000.0)
            
#             bone_loss = 0
#             bone_count = 0
#             for (j1, j2), expected_length in self.avg_bone_lengths.items():
#                 if keypoints_2d[j1, 2] > 0.3 and keypoints_2d[j2, 2] > 0.3:
#                     bone_length = np.linalg.norm(pose_3d[j1] - pose_3d[j2])
#                     bone_loss += ((bone_length - expected_length) / expected_length) ** 2
#                     bone_count += 1
#             if bone_count > 0:
#                 bone_loss /= bone_count
            
#             ankle_loss = 0
#             ankle_loss += np.sum((pose_3d[15, :2] - ankle_world[0]) ** 2) * 10
#             ankle_loss += np.sum((pose_3d[16, :2] - ankle_world[1]) ** 2) * 10
#             ankle_loss += (pose_3d[15, 2] - 0.04) ** 2 * 10
#             ankle_loss += (pose_3d[16, 2] - 0.04) ** 2 * 10
            
#             depth_loss = 0
#             for i in range(17):
#                 if pose_3d[i, 2] < 0:
#                     depth_loss += pose_3d[i, 2] ** 2 * 100
            
#             total_loss = reprojection_loss + 0.3 * bone_loss + ankle_loss + depth_loss
#             return total_loss
        
#         x0 = pose_3d_init.flatten()
        
#         bounds = []
#         for i in range(17):
#             bounds.append((-10, 20))
#             bounds.append((-10, 20))
#             bounds.append((0.0, 2.5))
        
#         result = minimize(loss_function, x0, method='L-BFGS-B', bounds=bounds,
#                          options={'maxiter': 200, 'ftol': 1e-4, 'maxfun': 500})
        
#         pose_3d_refined = result.x.reshape(17, 3)
#         return pose_3d_refined
    
#     def estimate_pose(self, keypoints_2d, ankle_world, frame_id=None, player_id=None):
#         keypoints_array = np.array(keypoints_2d)
#         ankle_world_coords = [[ankle_world[0]['world_x'], ankle_world[0]['world_y']],
#                               [ankle_world[1]['world_x'], ankle_world[1]['world_y']]]
        
#         focal_length, scale, cx, cy = self.estimate_camera_params(keypoints_array, ankle_world_coords)
        
#         if focal_length is None:
#             print(f"  [WARN] Frame {frame_id}, {player_id}: Could not estimate camera params")
#             return None
        
#         pose_3d_init = self.initialize_3d_pose(keypoints_array, ankle_world_coords, 
#                                                focal_length, scale, cx, cy)
        
#         pose_3d_refined = self.refine_3d_pose(pose_3d_init, keypoints_array, ankle_world_coords,
#                                               focal_length, scale, cx, cy)
        
#         return pose_3d_refined

# def process_json_file(json_path, output_path):
#     print("="*60)
#     print("3D POSE ESTIMATION - STARTING")
#     print("="*60)
#     print(f"Input file: {json_path}")
#     print(f"Output file: {output_path}")
#     print()
    
#     print("Loading JSON data...")
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     print(f"✓ Loaded successfully")
#     print(f"  Video: {data['video_info']['video_name']}")
#     print(f"  Frames: {data['video_info']['frame_count']}")
#     print(f"  Resolution: {data['video_info']['width']}x{data['video_info']['height']}")
#     print(f"  Court: {data['court_info']['width_meters']}m x {data['court_info']['length_meters']}m")
#     print()
    
#     estimator = PoseEstimator3D()
    
#     results = {
#         'video_info': data['video_info'],
#         'court_info': data['court_info'],
#         'poses_3d': {}
#     }
    
#     total_frames = len(data['frame_data'])
#     print(f"Processing {total_frames} frames...")
#     print("-"*60)
    
#     processed_count = 0
#     failed_count = 0
    
#     for idx, (frame_id, frame_data) in enumerate(data['frame_data'].items(), 1):
#         if idx % 50 == 0 or idx == 1:
#             print(f"Frame {idx}/{total_frames} (ID: {frame_id})")
        
#         frame_start_time = time.time()
#         results['poses_3d'][frame_id] = {}
        
#         for player_id, player_data in frame_data.items():
#             if 'keypoints_2d' not in player_data or 'ankles' not in player_data:
#                 if idx == 1:
#                     print(f"  [SKIP] Frame {frame_id}, {player_id}: Missing keypoints or ankle data")
#                 failed_count += 1
#                 continue
            
#             if len(player_data['ankles']) < 2:
#                 failed_count += 1
#                 continue
                
#             keypoints_2d = []
#             for kpt in player_data['keypoints_2d']:
#                 keypoints_2d.append([kpt['x'], kpt['y'], kpt['confidence']])
            
#             if len(keypoints_2d) != 17:
#                 failed_count += 1
#                 continue
            
#             pose_3d = estimator.estimate_pose(keypoints_2d, player_data['ankles'], 
#                                              frame_id, player_id)
            
#             if pose_3d is not None:
#                 results['poses_3d'][frame_id][player_id] = {
#                     'joints_3d': pose_3d.tolist(),
#                     'joint_names': [
#                         'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
#                         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
#                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
#                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
#                     ]
#                 }
#                 processed_count += 1
#             else:
#                 failed_count += 1
        
#         frame_time = time.time() - frame_start_time
#         if idx == 1:
#             print(f"  First frame took {frame_time:.2f}s")
#             estimated_total_time = frame_time * total_frames
#             print(f"  Estimated total time: {estimated_total_time/60:.1f} minutes")
    
#     print("-"*60)
#     print()
#     print("SUMMARY")
#     print("-"*60)
#     print(f"✓ Total frames processed: {total_frames}")
#     print(f"✓ Successful pose estimations: {processed_count}")
#     if failed_count > 0:
#         print(f"⚠ Failed estimations: {failed_count}")
#     print()
    
#     print(f"Saving results to {output_path}...")
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)
    
#     print(f"✓ Results saved successfully!")
#     print("="*60)
#     print("COMPLETE")
#     print("="*60)

# if __name__ == "__main__":
#     input_file = "positions.json"
#     output_file = "poses_3d_output.json"
    
#     process_json_file(input_file, output_file)


import numpy as np
import json
import torch
import urllib.request
import os

class MotionBERTPoseEstimator:
    def __init__(self, model_path=None):
        print("="*60)
        print("MOTIONBERT 3D POSE ESTIMATION")
        print("="*60)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.load_model(model_path)
        else:
            print("Model not found. Using lightweight estimation method.")
            self.model = None
        
        self.previous_poses = {}
        print()
    
    def load_model(self, model_path):
        try:
            from lib.model.DSTformer import DSTformer
            
            self.model = DSTformer(
                dim_in=3,
                dim_out=3,
                dim_feat=256,
                dim_rep=512,
                depth=5,
                num_heads=8,
                mlp_ratio=4,
                maxlen=243,
                num_joints=17
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_pos' in checkpoint:
                self.model.load_state_dict(checkpoint['model_pos'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print("✓ MotionBERT model loaded successfully")
            
        except Exception as e:
            print(f"⚠ Could not load MotionBERT: {e}")
            print("Falling back to lightweight method")
            self.model = None
    
    def normalize_2d_pose(self, keypoints_2d):
        valid_mask = keypoints_2d[:, 2] > 0.3
        if not np.any(valid_mask):
            return keypoints_2d, 1.0, np.array([960, 540])
        
        valid_points = keypoints_2d[valid_mask, :2]
        center = np.mean(valid_points, axis=0)
        
        scale = np.max(np.linalg.norm(valid_points - center, axis=1))
        if scale < 1e-6:
            scale = 1.0
        
        normalized = keypoints_2d.copy()
        normalized[:, :2] = (keypoints_2d[:, :2] - center) / scale
        
        return normalized, scale, center
    
    def estimate_with_model(self, keypoints_2d_batch):
        with torch.no_grad():
            input_tensor = torch.from_numpy(keypoints_2d_batch).float().to(self.device)
            output = self.model(input_tensor)
            pose_3d = output.cpu().numpy()
        
        return pose_3d[0]
    
    def estimate_lightweight(self, keypoints_2d, ankle_world):
        pose_3d = np.zeros((17, 3))
        
        pose_3d[15] = [ankle_world[0][0], ankle_world[0][1], 0.04]
        pose_3d[16] = [ankle_world[1][0], ankle_world[1][1], 0.04]
        
        ankle_center_world = (pose_3d[15, :2] + pose_3d[16, :2]) / 2
        ankle_center_2d = (keypoints_2d[15, :2] + keypoints_2d[16, :2]) / 2
        
        body_height_map = {
            0: 1.70, 1: 1.68, 2: 1.68, 3: 1.66, 4: 1.66,
            5: 1.45, 6: 1.45, 7: 1.20, 8: 1.20, 9: 1.05, 10: 1.05,
            11: 0.95, 12: 0.95, 13: 0.50, 14: 0.50, 15: 0.04, 16: 0.04
        }
        
        scale_factor = 0.003
        
        for i in range(17):
            if i not in [15, 16] and keypoints_2d[i, 2] > 0.3:
                dx = keypoints_2d[i, 0] - ankle_center_2d[0]
                dy = keypoints_2d[i, 1] - ankle_center_2d[1]
                
                z = body_height_map[i]
                x = ankle_center_world[0] + dx * scale_factor * z
                y = ankle_center_world[1] + dy * scale_factor * z
                
                pose_3d[i] = [x, y, z]
            elif i not in [15, 16]:
                pose_3d[i] = [ankle_center_world[0], ankle_center_world[1], body_height_map[i]]
        
        return pose_3d
    
    def apply_ankle_constraints(self, pose_3d, ankle_world):
        pose_3d[15, :2] = ankle_world[0]
        pose_3d[15, 2] = 0.04
        
        pose_3d[16, :2] = ankle_world[1]
        pose_3d[16, 2] = 0.04
        
        return pose_3d
    
    def temporal_smoothing(self, pose_3d, player_id, alpha=0.4):
        if player_id in self.previous_poses:
            prev_pose = self.previous_poses[player_id]
            smoothed = alpha * pose_3d + (1 - alpha) * prev_pose
        else:
            smoothed = pose_3d
        
        self.previous_poses[player_id] = pose_3d.copy()
        return smoothed
    
    def estimate_pose(self, keypoints_2d, ankle_world, frame_id=None, player_id=None):
        keypoints_array = np.array(keypoints_2d)
        ankle_world_coords = np.array([
            [ankle_world[0]['world_x'], ankle_world[0]['world_y']],
            [ankle_world[1]['world_x'], ankle_world[1]['world_y']]
        ])
        
        if self.model is not None:
            normalized, scale, center = self.normalize_2d_pose(keypoints_array)
            
            input_batch = normalized[:, :3].reshape(1, 1, 17, 3)
            
            pose_3d = self.estimate_with_model(input_batch)
            
            pose_3d = self.apply_ankle_constraints(pose_3d, ankle_world_coords)
        else:
            pose_3d = self.estimate_lightweight(keypoints_array, ankle_world_coords)
        
        pose_3d_smoothed = self.temporal_smoothing(pose_3d, player_id, alpha=0.3)
        
        return pose_3d_smoothed

def process_json_file(json_path, output_path, model_path=None):
    print("="*60)
    print("3D POSE ESTIMATION WITH AI MODEL")
    print("="*60)
    print(f"Input file: {json_path}")
    print(f"Output file: {output_path}")
    print()
    
    print("Loading JSON data...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded successfully")
    print(f"  Video: {data['video_info']['video_name']}")
    print(f"  Frames: {data['video_info']['frame_count']}")
    print(f"  Resolution: {data['video_info']['width']}x{data['video_info']['height']}")
    print(f"  Court: {data['court_info']['width_meters']}m x {data['court_info']['length_meters']}m")
    print()
    
    estimator = MotionBERTPoseEstimator(model_path)
    
    results = {
        'video_info': data['video_info'],
        'court_info': data['court_info'],
        'poses_3d': {}
    }
    
    total_frames = len(data['frame_data'])
    print(f"Processing {total_frames} frames...")
    print("-"*60)
    
    processed_count = 0
    failed_count = 0
    
    import time
    start_time = time.time()
    
    for idx, (frame_id, frame_data) in enumerate(data['frame_data'].items(), 1):
        if idx % 50 == 0 or idx == 1:
            elapsed = time.time() - start_time
            fps = idx / elapsed if elapsed > 0 else 0
            print(f"Frame {idx}/{total_frames} (ID: {frame_id}) - {fps:.1f} fps")
        
        results['poses_3d'][frame_id] = {}
        
        for player_id, player_data in frame_data.items():
            if 'keypoints_2d' not in player_data or 'ankles' not in player_data:
                failed_count += 1
                continue
            
            if len(player_data['ankles']) < 2:
                failed_count += 1
                continue
            
            keypoints_2d = []
            for kpt in player_data['keypoints_2d']:
                keypoints_2d.append([kpt['x'], kpt['y'], kpt['confidence']])
            
            if len(keypoints_2d) != 17:
                failed_count += 1
                continue
            
            try:
                pose_3d = estimator.estimate_pose(
                    keypoints_2d, 
                    player_data['ankles'],
                    frame_id, 
                    player_id
                )
                
                if pose_3d is not None:
                    results['poses_3d'][frame_id][player_id] = {
                        'joints_3d': pose_3d.tolist(),
                        'joint_names': [
                            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                        ]
                    }
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                if idx == 1:
                    print(f"  [ERROR] Frame {frame_id}, {player_id}: {e}")
                failed_count += 1
    
    elapsed = time.time() - start_time
    
    print("-"*60)
    print()
    print("SUMMARY")
    print("-"*60)
    print(f"✓ Total frames processed: {total_frames}")
    print(f"✓ Successful pose estimations: {processed_count}")
    print(f"✓ Processing time: {elapsed:.2f}s ({total_frames/elapsed:.1f} fps)")
    if failed_count > 0:
        print(f"⚠ Failed estimations: {failed_count}")
    print()
    
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved successfully!")
    print("="*60)
    print("COMPLETE")
    print("="*60)

if __name__ == "__main__":
    input_file = "positions.json"
    output_file = "poses_3d_output.json"
    
    model_path = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m/best_epoch.bin"
    
    if not os.path.exists(model_path):
        print()
        print("="*60)
        print("OPTIONAL: DOWNLOAD MOTIONBERT MODEL")
        print("="*60)
        print("For best results, download the pretrained model:")
        print("1. Download from: https://1drv.ms/u/s!AvAdh0LSjEOlcfmhYSYcxz46rL8")
        print("2. Save to: checkpoint/pose3d/FT_MB_lite_MB_ft_h36m/best_epoch.bin")
        print()
        print("Without the model, using lightweight geometric estimation.")
        print("="*60)
        print()
        model_path = None
    
    process_json_file(input_file, output_file, model_path)