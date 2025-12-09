import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

class Pose3DViewer:
    def __init__(self):
        self.coco_skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [13, 15], [12, 14], [14, 16]
        ]
        
        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        self.colors = {
            'player_0': '#00FF00',
            'player_1': '#FF00FF'
        }
    
    def load_poses(self, poses_path):
        print("="*60)
        print("LOADING 3D POSES")
        print("="*60)
        print(f"File: {poses_path}")
        
        with open(poses_path, 'r') as f:
            data = json.load(f)
        
        self.court_info = data['court_info']
        self.poses_3d = data['poses_3d']
        
        print(f"✓ Loaded {len(self.poses_3d)} frames")
        print(f"  Court: {self.court_info['width_meters']}m x {self.court_info['length_meters']}m")
        print()
        
        return data
    
    def draw_court(self, ax):
        width = self.court_info['width_meters']
        length = self.court_info['length_meters']
        
        court_x = [0, width, width, 0, 0]
        court_y = [0, 0, length, length, 0]
        court_z = [0, 0, 0, 0, 0]
        
        ax.plot(court_x, court_y, court_z, 'b-', linewidth=2, alpha=0.5)
        
        mid_line_y = length / 2
        ax.plot([0, width], [mid_line_y, mid_line_y], [0, 0], 'b--', linewidth=1, alpha=0.3)
        
        service_line_1 = length * 0.3
        service_line_2 = length * 0.7
        ax.plot([0, width], [service_line_1, service_line_1], [0, 0], 'b--', linewidth=1, alpha=0.3)
        ax.plot([0, width], [service_line_2, service_line_2], [0, 0], 'b--', linewidth=1, alpha=0.3)
        
        ax.text(width/2, -0.5, 0, 'Badminton Court', fontsize=10, ha='center')
    
    def draw_pose(self, ax, pose_3d, color, alpha=1.0):
        for bone in self.coco_skeleton:
            j1, j2 = bone
            x = [pose_3d[j1][0], pose_3d[j2][0]]
            y = [pose_3d[j1][1], pose_3d[j2][1]]
            z = [pose_3d[j1][2], pose_3d[j2][2]]
            ax.plot(x, y, z, color=color, linewidth=3, alpha=alpha)
        
        xs = pose_3d[:, 0]
        ys = pose_3d[:, 1]
        zs = pose_3d[:, 2]
        ax.scatter(xs, ys, zs, c=color, s=50, alpha=alpha, edgecolors='white', linewidth=1)
        
        ankle_indices = [15, 16]
        ax.scatter(pose_3d[ankle_indices, 0], 
                  pose_3d[ankle_indices, 1], 
                  pose_3d[ankle_indices, 2], 
                  c='red', s=100, alpha=alpha, edgecolors='yellow', linewidth=2, marker='D')
    
    def visualize_single_frame(self, frame_id):
        print(f"Visualizing frame {frame_id}...")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        self.draw_court(ax)
        
        if str(frame_id) in self.poses_3d:
            frame_data = self.poses_3d[str(frame_id)]
            
            for player_id, player_data in frame_data.items():
                if 'joints_3d' in player_data:
                    pose_3d = np.array(player_data['joints_3d'])
                    color = self.colors.get(player_id, '#FFFFFF')
                    self.draw_pose(ax, pose_3d, color)
        
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        ax.set_zlabel('Z (meters)', fontsize=10)
        ax.set_title(f'3D Pose Visualization - Frame {frame_id}', fontsize=14, fontweight='bold')
        
        ax.set_xlim(-1, self.court_info['width_meters'] + 1)
        ax.set_ylim(-1, self.court_info['length_meters'] + 1)
        ax.set_zlim(0, 2.5)
        
        ax.view_init(elev=20, azim=45)
        
        legend_elements = [
            mpatches.Patch(color=self.colors['player_0'], label='Player 0'),
            mpatches.Patch(color=self.colors['player_1'], label='Player 1'),
            mpatches.Patch(color='red', label='Ankles (Ground Truth)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Visualization complete")
    
    def visualize_animation(self, start_frame=1, end_frame=None, interval=100, save_path=None):
        print("="*60)
        print("CREATING 3D ANIMATION")
        print("="*60)
        
        frame_ids = sorted([int(k) for k in self.poses_3d.keys()])
        
        if end_frame is None:
            end_frame = frame_ids[-1]
        
        frame_ids = [f for f in frame_ids if start_frame <= f <= end_frame]
        
        print(f"Frames: {start_frame} to {end_frame}")
        print(f"Total frames: {len(frame_ids)}")
        print()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def init():
            ax.clear()
            return []
        
        def update(frame_idx):
            ax.clear()
            
            self.draw_court(ax)
            
            frame_id = frame_ids[frame_idx]
            
            if str(frame_id) in self.poses_3d:
                frame_data = self.poses_3d[str(frame_id)]
                
                for player_id, player_data in frame_data.items():
                    if 'joints_3d' in player_data:
                        pose_3d = np.array(player_data['joints_3d'])
                        color = self.colors.get(player_id, '#FFFFFF')
                        self.draw_pose(ax, pose_3d, color)
            
            ax.set_xlabel('X (meters)', fontsize=10)
            ax.set_ylabel('Y (meters)', fontsize=10)
            ax.set_zlabel('Z (meters)', fontsize=10)
            ax.set_title(f'3D Pose Animation - Frame {frame_id}/{end_frame}', 
                        fontsize=14, fontweight='bold')
            
            ax.set_xlim(-1, self.court_info['width_meters'] + 1)
            ax.set_ylim(-1, self.court_info['length_meters'] + 1)
            ax.set_zlim(0, 2.5)
            
            ax.view_init(elev=20, azim=45)
            
            legend_elements = [
                mpatches.Patch(color=self.colors['player_0'], label='Player 0'),
                mpatches.Patch(color=self.colors['player_1'], label='Player 1'),
                mpatches.Patch(color='red', label='Ankles')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            if frame_idx % 10 == 0:
                print(f"  Rendering frame {frame_idx}/{len(frame_ids)}")
            
            return []
        
        anim = FuncAnimation(fig, update, init_func=init, 
                           frames=len(frame_ids), interval=interval, 
                           blit=False, repeat=True)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=10)
            print(f"✓ Animation saved")
        else:
            print("Displaying animation... (Close window to exit)")
            plt.tight_layout()
            plt.show()
        
        print("="*60)
        print("COMPLETE")
        print("="*60)
    
    def visualize_multiple_frames(self, frame_ids, rows=2, cols=3):
        print(f"Visualizing {len(frame_ids)} frames...")
        
        fig = plt.figure(figsize=(18, 12))
        
        for idx, frame_id in enumerate(frame_ids[:rows*cols]):
            ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
            
            self.draw_court(ax)
            
            if str(frame_id) in self.poses_3d:
                frame_data = self.poses_3d[str(frame_id)]
                
                for player_id, player_data in frame_data.items():
                    if 'joints_3d' in player_data:
                        pose_3d = np.array(player_data['joints_3d'])
                        color = self.colors.get(player_id, '#FFFFFF')
                        self.draw_pose(ax, pose_3d, color, alpha=0.8)
            
            ax.set_xlabel('X (m)', fontsize=8)
            ax.set_ylabel('Y (m)', fontsize=8)
            ax.set_zlabel('Z (m)', fontsize=8)
            ax.set_title(f'Frame {frame_id}', fontsize=10)
            
            ax.set_xlim(-1, self.court_info['width_meters'] + 1)
            ax.set_ylim(-1, self.court_info['length_meters'] + 1)
            ax.set_zlim(0, 2.5)
            
            ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Visualization complete")

if __name__ == "__main__":
    viewer = Pose3DViewer()
    viewer.load_poses("poses_3d_output.json")
    
    print("Choose visualization mode:")
    print("1. Single frame")
    print("2. Multiple frames grid")
    print("3. Animation (interactive)")
    print("4. Animation (save as GIF)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        frame_id = int(input("Enter frame ID: "))
        viewer.visualize_single_frame(frame_id)
    
    elif choice == "2":
        start = int(input("Enter start frame: "))
        end = int(input("Enter end frame: "))
        step = int(input("Enter step (e.g., 50): "))
        frames = list(range(start, end+1, step))
        viewer.visualize_multiple_frames(frames)
    
    elif choice == "3":
        start = int(input("Enter start frame (default 1): ") or "1")
        end = int(input("Enter end frame (default last): ") or "0")
        end = None if end == 0 else end
        viewer.visualize_animation(start, end)
    
    elif choice == "4":
        start = int(input("Enter start frame: "))
        end = int(input("Enter end frame: "))
        output = input("Enter output filename (e.g., animation.gif): ").strip()
        if not output.endswith('.gif'):
            output += '.gif'
        viewer.visualize_animation(start, end, save_path=output)
    
    else:
        print("Invalid choice, showing frame 1")
        viewer.visualize_single_frame(1)