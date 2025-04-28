# YOLO + Pose Estimation LSTM
  
  
## Dataset generate
Yolo 11m -> People Detect -> MediaPipe -> Pose landmark  
=> LSTM training dataset

```
python generate_json_multi.py
```
### Sources of the datasets
Kranok-NV  
https://www.kaggle.com/datasets/kevinbkwanloo/kranoknv
  
  
## LSTM training
```
python LSTM.py
```
![poster](./training_curves.png)