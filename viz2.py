import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class Advanced3DViewer:
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
        print("LOADING 3D POSES FOR VISUALIZATION")
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
    
    def create_court_mesh(self):
        width = self.court_info['width_meters']
        length = self.court_info['length_meters']
        
        court_x = [0, width, width, 0, 0, width, width, 0]
        court_y = [0, 0, length, length, 0, 0, length, length]
        court_z = [0, 0, 0, 0, 0, 0, 0, 0]
        
        i = [0, 0, 0, 1, 4, 4]
        j = [1, 2, 3, 2, 5, 6]
        k = [2, 3, 1, 3, 6, 7]
        
        court_mesh = go.Mesh3d(
            x=court_x, y=court_y, z=court_z,
            i=i, j=j, k=k,
            color='lightblue',
            opacity=0.3,
            name='Court',
            showlegend=True
        )
        
        lines = []
        
        boundary = [[0, width], [0, 0], [0, 0]]
        lines.append(go.Scatter3d(x=boundary[0], y=boundary[1], z=boundary[2],
                                 mode='lines', line=dict(color='blue', width=4),
                                 showlegend=False))
        
        boundary = [[width, width], [0, length], [0, 0]]
        lines.append(go.Scatter3d(x=boundary[0], y=boundary[1], z=boundary[2],
                                 mode='lines', line=dict(color='blue', width=4),
                                 showlegend=False))
        
        boundary = [[width, 0], [length, length], [0, 0]]
        lines.append(go.Scatter3d(x=boundary[0], y=boundary[1], z=boundary[2],
                                 mode='lines', line=dict(color='blue', width=4),
                                 showlegend=False))
        
        boundary = [[0, 0], [length, 0], [0, 0]]
        lines.append(go.Scatter3d(x=boundary[0], y=boundary[1], z=boundary[2],
                                 mode='lines', line=dict(color='blue', width=4),
                                 showlegend=False))
        
        mid_y = length / 2
        lines.append(go.Scatter3d(x=[0, width], y=[mid_y, mid_y], z=[0, 0],
                                 mode='lines', line=dict(color='blue', width=2, dash='dash'),
                                 showlegend=False))
        
        return [court_mesh] + lines
    
    def create_skeleton_tubes(self, pose_3d, color, player_name):
        traces = []
        
        for bone_idx, (j1, j2) in enumerate(self.coco_skeleton):
            x = [pose_3d[j1][0], pose_3d[j2][0]]
            y = [pose_3d[j1][1], pose_3d[j2][1]]
            z = [pose_3d[j1][2], pose_3d[j2][2]]
            
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=8),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        xs = [p[0] for p in pose_3d]
        ys = [p[1] for p in pose_3d]
        zs = [p[2] for p in pose_3d]
        
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=6, color=color, 
                       line=dict(color='white', width=2)),
            name=player_name,
            text=self.joint_names,
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m',
        ))
        
        ankle_indices = [15, 16]
        ankle_x = [pose_3d[i][0] for i in ankle_indices]
        ankle_y = [pose_3d[i][1] for i in ankle_indices]
        ankle_z = [pose_3d[i][2] for i in ankle_indices]
        
        traces.append(go.Scatter3d(
            x=ankle_x, y=ankle_y, z=ankle_z,
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond',
                       line=dict(color='yellow', width=3)),
            name=f'{player_name} Ankles',
            text=['Left Ankle', 'Right Ankle'],
            hovertemplate='<b>%{text}</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m',
        ))
        
        return traces
    
    def visualize_single_frame_interactive(self, frame_id):
        print(f"Creating interactive visualization for frame {frame_id}...")
        
        fig = go.Figure()
        
        court_traces = self.create_court_mesh()
        for trace in court_traces:
            fig.add_trace(trace)
        
        if str(frame_id) in self.poses_3d:
            frame_data = self.poses_3d[str(frame_id)]
            
            for player_id, player_data in frame_data.items():
                if 'joints_3d' in player_data:
                    pose_3d = np.array(player_data['joints_3d'])
                    color = self.colors.get(player_id, '#FFFFFF')
                    player_name = player_id.replace('_', ' ').title()
                    
                    skeleton_traces = self.create_skeleton_tubes(pose_3d, color, player_name)
                    for trace in skeleton_traces:
                        fig.add_trace(trace)
        
        fig.update_layout(
            title=dict(
                text=f'3D Pose Reconstruction - Frame {frame_id}',
                font=dict(size=20, color='white')
            ),
            scene=dict(
                xaxis=dict(title='X (meters)', backgroundcolor="rgb(230, 230,230)",
                          gridcolor="white", showbackground=True, range=[-1, self.court_info['width_meters']+1]),
                yaxis=dict(title='Y (meters)', backgroundcolor="rgb(230, 230,230)",
                          gridcolor="white", showbackground=True, range=[-1, self.court_info['length_meters']+1]),
                zaxis=dict(title='Z (meters)', backgroundcolor="rgb(230, 230,230)",
                          gridcolor="white", showbackground=True, range=[0, 2.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2.2, z=0.4),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            showlegend=True,
            legend=dict(x=0.7, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
            width=1200,
            height=800,
            paper_bgcolor='rgb(243, 243, 243)',
            plot_bgcolor='rgb(243, 243, 243)'
        )
        
        fig.show()
        print("✓ Interactive visualization opened in browser")
    
    def create_animation(self, start_frame=1, end_frame=None, step=1, output_html=None):
        print("="*60)
        print("CREATING 3D ANIMATION")
        print("="*60)
        
        frame_ids = sorted([int(k) for k in self.poses_3d.keys()])
        
        if end_frame is None:
            end_frame = frame_ids[-1]
        
        frame_ids = [f for f in frame_ids if start_frame <= f <= end_frame][::step]
        
        print(f"Frames: {start_frame} to {end_frame} (step: {step})")
        print(f"Total frames: {len(frame_ids)}")
        print()
        
        frames = []
        
        for idx, frame_id in enumerate(frame_ids):
            frame_traces = []
            
            court_traces = self.create_court_mesh()
            frame_traces.extend(court_traces)
            
            if str(frame_id) in self.poses_3d:
                frame_data = self.poses_3d[str(frame_id)]
                
                for player_id, player_data in frame_data.items():
                    if 'joints_3d' in player_data:
                        pose_3d = np.array(player_data['joints_3d'])
                        color = self.colors.get(player_id, '#FFFFFF')
                        player_name = player_id.replace('_', ' ').title()
                        
                        skeleton_traces = self.create_skeleton_tubes(pose_3d, color, player_name)
                        frame_traces.extend(skeleton_traces)
            
            frames.append(go.Frame(data=frame_traces, name=str(frame_id)))
            
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(frame_ids)} frames")
        
        initial_traces = frames[0].data if frames else []
        
        fig = go.Figure(data=initial_traces, frames=frames)
        
        fig.update_layout(
            title=dict(
                text='3D Pose Animation',
                font=dict(size=20, color='white')
            ),
            scene=dict(
                xaxis=dict(title='X (meters)', backgroundcolor="rgb(230, 230,230)",
                          gridcolor="white", showbackground=True, range=[-1, self.court_info['width_meters']+1]),
                yaxis=dict(title='Y (meters)', backgroundcolor="rgb(230, 230,230)",
                          gridcolor="white", showbackground=True, range=[-1, self.court_info['length_meters']+1]),
                zaxis=dict(title='Z (meters)', backgroundcolor="rgb(230, 230,230)",
                          gridcolor="white", showbackground=True, range=[0, 2.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2.2, z=0.4),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                    'fromcurrent': True, 'mode': 'immediate'}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate'}]}
                ],
                'x': 0.1, 'y': 0.95
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'y': 0.02,
                'xanchor': 'left',
                'x': 0.1,
                'len': 0.85,
                'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                               'mode': 'immediate'}],
                          'label': f.name, 'method': 'animate'}
                         for f in frames]
            }],
            showlegend=True,
            legend=dict(x=0.7, y=0.95, bgcolor='rgba(255,255,255,0.8)'),
            width=1200,
            height=800,
            paper_bgcolor='rgb(243, 243, 243)',
            plot_bgcolor='rgb(243, 243, 243)'
        )
        
        if output_html:
            fig.write_html(output_html)
            print(f"✓ Animation saved to {output_html}")
            print(f"  Open in browser to view interactive 3D animation")
        else:
            fig.show()
            print("✓ Animation opened in browser")
        
        print("="*60)
        print("COMPLETE")
        print("="*60)

if __name__ == "__main__":
    viewer = Advanced3DViewer()
    viewer.load_poses("poses_3d_output.json")
    
    print("Choose visualization mode:")
    print("1. Single frame (Interactive 3D)")
    print("2. Animation (Interactive HTML)")
    print("3. Animation (Save HTML file)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        frame_id = int(input("Enter frame ID: "))
        viewer.visualize_single_frame_interactive(frame_id)
    
    elif choice == "2":
        start = int(input("Enter start frame (default 1): ") or "1")
        end = int(input("Enter end frame (default last): ") or "0")
        step = int(input("Enter step (default 1): ") or "1")
        end = None if end == 0 else end
        viewer.create_animation(start, end, step)
    
    elif choice == "3":
        start = int(input("Enter start frame: "))
        end = int(input("Enter end frame: "))
        step = int(input("Enter step (e.g., 2 for every 2nd frame): "))
        output = input("Enter output filename (e.g., demo.html): ")
        if not output.endswith('.html'):
            output += '.html'
        viewer.create_animation(start, end, step, output_html=output)
    
    else:
        print("Invalid choice, showing frame 1")
        viewer.visualize_single_frame_interactive(1)