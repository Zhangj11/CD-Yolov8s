需要会使用基本yolov8s源码，只需将对应文件放入yolov8s中即可

将yaml为改进的yolov8s的配置文件,可以放入ultralytics/cfg/models/下
CCSM.py，RFMDSConv，We_Concat需要放到ultralytics/nn路径下
再修改tasks.py文件中的import，以及671行代码将对应模块写入 if m in (...)中即可
比如from ultralytics.nn.RFCCSM import RFCCSMConv

就可以直接运行训练代码，需要将训练的数据集路径和配置文件路径修改
训练代码：yolo detect train data=datasets/drone/drone.yaml model=ultralytics/cfg/models/new/try/v8-RFMDS-CCMS2-We_Concat.yaml pretrained=yolov8s.pt
