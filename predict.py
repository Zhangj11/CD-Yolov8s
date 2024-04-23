from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/v8-RFMDS-CCMS2-bifpn/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(source='datasets/pre/', save=True, name='result/pre-result1')
# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk