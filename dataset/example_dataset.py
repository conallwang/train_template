import json
import os

import numpy as np
from torch.utils.data import Dataset


class ExampleData(Dataset):
    def __init__(self, config, split="train"):
        super().__init__()

        self.config = config
        self.img_h, self.img_w = config["data.img_h"], config["data.img_w"]
        self.rate_h, self.rate_w = self.img_h / 802.0, self.img_w / 550.0

        self.basedir = config["data.root"]

        self.datapaths = []
        # TODO: add codes to load data path or pre-load data

    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, index):
        datapth = self.datapaths[index]

        sample = {}
        # TODO: add codes to load all data needed

        return sample
