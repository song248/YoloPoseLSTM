# YOLO + Pose Estimation LSTM
  
  
## Dataset generate
Yolo 11m -> People Detect -> MediaPipe -> Pose landmark  
=> LSTM training dataset

```
python generate_json_multi.py
```

```
self.coordinate_for_angel = [
            [8, 6, 2],      # right shoulder - right elbow - right wrist
            [11, 5, 7],     # left shoulder - left elbow - left wrist
            [6, 8, 10],     # right elbow - right wrist - right hand
            [5, 7, 9],      # left elbow - left wrist - left hand
            [6, 12, 14],    # right elbow - right hip - right knee
            [5, 11, 13],    # left elbow - left hip - left knee
            [12, 14, 16],   # right hip - right knee - right ankle
            [11, 13, 15]    # left hip - left knee - left ankle
        ]
```

### Sources of the datasets
Kranok-NV  
https://www.kaggle.com/datasets/kevinbkwanloo/kranoknv
  
  
## LSTM training
```
python LSTM.py
```
![poster](./training_curves.png)