import json
from PIL import Image
import numpy as np
import cv2

from deeplab import DeeplabV3


def init():
    deeplab = DeeplabV3()
    return deeplab

def process_image(handle=None,input_image=None,args=None, **kwargs):
    name_classes = ["background","algae","dead_twigs_leaves","garbage","water"]

    args =json.loads(args)
    mask_output_path =args['mask_output_path']

    # Process image here
    # Generate dummy mask data
    image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    pred_mask_per_frame = handle.detect_image(image, count=True, name_classes=name_classes)
    pred_mask_per_frame.save(mask_output_path)
    return json.dumps({'mask': mask_output_path}, indent=4)
