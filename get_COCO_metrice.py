import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='datasets/drone.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default='runs/detect/val13/predictions.json', help='data yaml path')

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
