import cv2
import numpy as np
import json

class PoseVisualizer:
    def __init__(self):
        self.coco_skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [13, 15], [12, 14], [14, 16]
        ]
        
        self.colors = {
            'player_0': (0, 255, 0),
            'player_1': (255, 0, 255),
            'ankle': (0, 0, 255),
            'joint': (255, 255, 0)
        }
        
        self.line_thickness = 3
        self.joint_radius = 5
        self.ankle_radius = 8
    
    def project_3d_to_2d(self, pose_3d, focal_length=1920, cx=960, cy=540):
        projected = []
        for joint in pose_3d:
            x, y, z = joint
            if z > 0.01:
                px = int((focal_length * x / z) + cx)
                py = int((focal_length * y / z) + cy)
                projected.append([px, py])
            else:
                projected.append([0, 0])
        return np.array(projected)
    
    def draw_skeleton(self, frame, joints_2d, color, is_ankle_indices=[15, 16]):
        h, w = frame.shape[:2]
        
        for i, (j1_idx, j2_idx) in enumerate(self.coco_skeleton):
            pt1 = joints_2d[j1_idx]
            pt2 = joints_2d[j2_idx]
            
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(frame, tuple(pt1), tuple(pt2), color, self.line_thickness)
        
        for i, joint in enumerate(joints_2d):
            if joint[0] > 0 and joint[1] > 0:
                if 0 <= joint[0] < w and 0 <= joint[1] < h:
                    if i in is_ankle_indices:
                        cv2.circle(frame, tuple(joint), self.ankle_radius, self.colors['ankle'], -1)
                        cv2.circle(frame, tuple(joint), self.ankle_radius + 2, color, 2)
                    else:
                        cv2.circle(frame, tuple(joint), self.joint_radius, color, -1)
        
        return frame
    
    def add_info_panel(self, frame, frame_id, player_id, pose_3d):
        h, w = frame.shape[:2]
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        y_offset = 30
        cv2.putText(panel, f"Frame: {frame_id}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(panel, f"Player: {player_id}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if pose_3d is not None and len(pose_3d) > 15:
            ankle_left = pose_3d[15]
            ankle_right = pose_3d[16]
            y_offset += 30
            cv2.putText(panel, f"Ankle L: ({ankle_left[0]:.2f}, {ankle_left[1]:.2f}, {ankle_left[2]:.2f})m", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            cv2.putText(panel, f"Ankle R: ({ankle_right[0]:.2f}, {ankle_right[1]:.2f}, {ankle_right[2]:.2f})m", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        combined = np.vstack([frame, panel])
        return combined
    
    def visualize_video(self, video_path, poses_3d_path, output_path, show_info=True):
        print("="*60)
        print("3D POSE VISUALIZATION")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"Poses: {poses_3d_path}")
        print(f"Output: {output_path}")
        print()
        
        with open(poses_3d_path, 'r') as f:
            data = json.load(f)
        
        poses_3d = data['poses_3d']
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if show_info:
            output_height = height + 120
        else:
            output_height = height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))
        
        print(f"Video Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print()
        print("Processing frames...")
        print("-"*60)
        
        frame_idx = 1
        processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id = str(frame_idx)
            
            if frame_id in poses_3d:
                frame_poses = poses_3d[frame_id]
                
                for player_id, player_data in frame_poses.items():
                    if 'joints_3d' in player_data:
                        pose_3d = np.array(player_data['joints_3d'])
                        joints_2d = self.project_3d_to_2d(pose_3d)
                        color = self.colors.get(player_id, (255, 255, 255))
                        frame = self.draw_skeleton(frame, joints_2d, color)
                
                if show_info and len(frame_poses) > 0:
                    first_player = list(frame_poses.keys())[0]
                    first_pose = np.array(frame_poses[first_player]['joints_3d'])
                    frame = self.add_info_panel(frame, frame_id, 
                                               f"{len(frame_poses)} players", first_pose)
                elif show_info:
                    frame = self.add_info_panel(frame, frame_id, "No pose", None)
            else:
                if show_info:
                    frame = self.add_info_panel(frame, frame_id, "No data", None)
            
            out.write(frame)
            processed += 1
            
            if processed % 50 == 0:
                print(f"Processed {processed}/{total_frames} frames")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print("-"*60)
        print()
        print("SUMMARY")
        print("-"*60)
        print(f"✓ Total frames processed: {processed}")
        print(f"✓ Output saved to: {output_path}")
        print("="*60)
        print("COMPLETE")
        print("="*60)

if __name__ == "__main__":
    visualizer = PoseVisualizer()
    
    video_path = "ds2.mp4"
    poses_3d_path = "poses_3d_output.json"
    output_path = "output_visualized.mp4"
    
    visualizer.visualize_video(video_path, poses_3d_path, output_path, show_info=True)