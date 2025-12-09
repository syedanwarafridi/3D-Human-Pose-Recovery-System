# 3D Human Pose Reconstruction from 2D COCO Keypoints (No Camera Calibration)

This project estimates **3D human pose** directly from **2D COCO keypoints**, using only ankle world-coordinates as reference — making it functional **without camera calibration**.  
The system supports both deep-learning inference (MotionBERT) and a fallback geometric estimator.

---

## 📌 Overview

This pipeline processes per-frame 2D detection data and outputs smoothed 3D joint coordinates in world space.  
It is designed for use in sports analytics or multi-angle recordings where camera parameters are unknown.

---

## 🔥 Core Pipeline

1. **Input Parsing**
   - Reads COCO 17-joint 2D keypoints from JSON  
   - Extracts ankle world-coordinates as the global reference anchor

2. **3D Pose Generation (Two-Stage Logic)**  
   - **Using MotionBERT (if available):**  
     - Normalizes 2D joints  
     - Lifts to 3D via pretrained MotionBERT  
     - Re-anchors pose using provided ankle world coordinates  
   - **Fallback Lightweight Estimator:**  
     - Reconstructs joint depth using biomechanical height mapping  
     - Requires no learned model or camera calibration  

3. **Spatial Constraints**
   - Ankles fixed in world space  
   - Body height ratios used to approximate depth  
   - Works across variable camera angles

4. **Temporal Smoothing**
   - Frame-to-frame blending reduces jitter  
   - Produces stable 3D motion output

---

## 📤 Output

The system exports structured JSON:

```json
{
  "poses_3d": {
    "frame_id": {
      "player_id": {
        "joints_3d": [...],
        "joint_names": [...]
      }
    }
  }
}
