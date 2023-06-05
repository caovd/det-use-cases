
#!/usr/bin/env python3

import os

from yolox.exp import Exp as MyExp

# NEW - Step 3 - hpsearch - add hparams
class Exp(MyExp):
    def __init__(self, hparams):
        super(Exp, self).__init__(hparams)
        self.hparams = hparams
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Define yourself dataset path
        self.data_dir = "./datasets/airbus"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"
        self.max_epoch = 5

        self.num_classes = 1
