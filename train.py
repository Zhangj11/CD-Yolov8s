from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("ultralytics/cfg/models/v8-RFMDS-CCMS2-We_Concat.yaml")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="datasets/drone/drone.yaml",
    epochs=200,
    imgsz=640,
    pretrained='yolov8s.pt',
    batch=8)
