from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

from ipg_reward import ImagePromptGenerationReward
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from transformers import AutoProcessor

# Use the images in COCO Caption dataset
class ImagePromptGenerationDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        annFile: str
    ):
        self.coco_dataset = datasets.CocoCaptions(
                                                root = './cocodata/val2017/',
                                                annFile = './cocodata/annotations/captions_val2017.json',
                                                #transform=transforms.Compose([transforms.Resize((224,224)), transforms.PILToTensor()]) # TODO: use small image size for now
                                                )# 
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, caption = self.coco_dataset[idx]
        image = self.processor(images=image, return_tensors="pt")
        #image['pixel_values'] = 
        return {'image': image['pixel_values'][0]}

def make_image_prompot_generation_dataset(config: "DictConfig") -> Tuple[ImagePromptGenerationDataset]:
    data_dict = {}
    for split in ['train']: # TODO: dev, test
        ipg_dataset = ImagePromptGenerationDataset(config.cocodataset_root, config.cocodataset_annFile)
        data_dict[split] = ipg_dataset
    return data_dict['train']

@dataclass
class ImagePromptGenerationDatasetConfig:
    cocodataset_root = './cocodata/val2017/'
    cocodataset_annFile = './cocodata/annotations/captions_val2017.json'


def make_image_prompt_generation_reward(
    num_classes: int,
    verbalizers: List[str],
    template: Optional[str],  
    config: "DictConfig") -> ImagePromptGenerationReward:
    return ImagePromptGenerationReward( config.task_lm, 
                                        config.is_mask_lm, 
                                        config.compute_zscore, 
                                        config.incorrect_coeff, 
                                        config.correct_coeff,
                                        num_classes, 
                                        verbalizers, 
                                        template)


@dataclass
class ImagePromptGenerationRewardConfig:
    task_lm: str = 'distilroberta-base'
    is_mask_lm: Optional[bool] = None
    compute_zscore: bool = True
    incorrect_coeff: float = 180.0
    correct_coeff: float = 200.0
