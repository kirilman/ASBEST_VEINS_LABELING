import torch
import torchvision
import sys
import pandas as pd
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



class Sam_processor:
    def __init__(self, 
                 sam_weight = "/storage/reshetnikov/sam/sam_vit_l_0b3195.pth",
                 model_type = "vit_l",
                 device = "cuda:2",
                 points_per_side = 24):
        sam = sam_model_registry[model_type](checkpoint=sam_weight)
        sam.to(device = device)
        mask_generator = SamAutomaticMaskGenerator(
                            model=sam,
                            points_per_side=points_per_side,
                            pred_iou_thresh=0.9,
                            stability_score_thresh=0.92,
                            crop_n_layers=1,
                            # crop_overlap_ratio = 0.5,
                            # crop_n_points_downscale_factor=2,
                            min_mask_region_area=1000,  # Requires open-cv to run post-processing
                        )
    
