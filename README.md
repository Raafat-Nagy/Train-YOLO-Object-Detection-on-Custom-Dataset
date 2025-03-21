# **Train YOLO11 for Person Detection Using Roboflow Dataset**

## **Overview**
This project demonstrates how to train a YOLO11 model for person detection using a dataset from Roboflow. The guide includes installing dependencies, downloading the dataset, updating configurations, training the model, and evaluating its performance.

---

## **Steps**

1. **Install Required Libraries**
   ```bash
   pip install ultralytics roboflow
   ```

2. **Download Dataset**
   - You can download the dataset from Roboflow directly using this link:  
   [Download Person Detection Dataset](https://universe.roboflow.com/titulacin/person-detection-9a6mk/dataset/16)  
   - Alternatively, you can use the Roboflow API to fetch the dataset programmatically.

3. **Dataset Split**
   - Train Set: 80% (4407 Images)  
   - Valid Set: 20% (1071 Images)  

4. **Train the Model**
   - Load YOLO11 and start training:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11s.pt')
   model.train(data='data.yaml', epochs=100, imgsz=640, patience=10)
   ```

5. **Evaluate the Model**
   - Validate the trained model:
   ```python
   metrics = model.val(data='data.yaml')
   ```

---

## **Model Weights**
You can download the trained model weights from the following link:  
[Download Weights](https://github.com/Raafat-Nagy/Train-YOLO-Object-Detection-on-Custom-Dataset/tree/main/runs/detect/train/weights)

---

## **Evaluation Results**
| Class | Images | Instances | Box(P) | R | mAP50 | mAP50-95 |
|-------|--------|-----------|--------|---|-------|----------|
| all   | 1071   | 2293      | 0.845  | 0.717 | 0.825 | 0.547    |

---

## **Key Metrics**
- mAP50: 82.5%
- mAP50-95: 54.7%
