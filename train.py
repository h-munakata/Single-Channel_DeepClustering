import sys
import torch
import yaml
from Learning import model
from Learning.trainer import Trainer
from Learning.dataloader import make_dataloader




def train():
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    dpcl = model.DeepClustering(config)

    train_dataloader, val_dataloader = make_dataloader(config)

    trainer = Trainer(train_dataloader, val_dataloader, dpcl, config)

    trainer.run()


if __name__ == "__main__":
    train()