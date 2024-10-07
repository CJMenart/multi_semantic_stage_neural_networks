"""
Dataset object that scrapes all images in folders within directory.
Currently being written to support inference on datasets we didn't train on.
"""
import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset
from PIL import Image
from pathlib import Path
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg

class RecursiveImageFolder(Dataset):

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        cache = False
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.images = []

        img_extensions = ['jpg', 'jpeg', 'png']
        for extension in img_extensions:
            for path in Path(self.root).rglob(f'*.{extension}'):
                self.images.append(path)
                        
        self.cache = cache
        if self.cache:
            self.imgcache = [None]*len(self.images)
                    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        
        if self.cache and self.imgcache[index]:
            image = self.imgcache[index]
        else:
            image = Image.open(str(self.images[index])).convert("RGB")
            if self.cache:
                self.imgcache[index] = image

        data_dict = {'Image': image}

        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        
        return data_dict

    def __len__(self) -> int:
        return len(self.images)

