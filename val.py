from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train2/weights/best.pt')  # load a custom model
# print("abcd")
# Validate the model
# 验证的时候把batch-size调成1
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(metrics.box)  # ap,ap50,ap75
