import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import jittor as jt
from utils.config import load_config, parse_args
from models.diffusion import DiffusionEngine
from dataset import DreamBoothDataset, collate_fn
from trainers.dreambooth_trainer import DreamBoothTrainer
from jittor.compatibility.utils.data import DataLoader
import argparse



def main():
    jt.flags.use_cuda = 1

    ### 1. Load Config
    args = parse_args()
    config = load_config(args.config)

    ### 2. Prepare Engine / Model
    engine = DiffusionEngine(config)

    ### 3. Prepare Dataset
    train_dataset = DreamBoothDataset(config.data, engine.tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.data.dataloader_num_workers
    )

    ### 4. Prepare Trainer and Train
    trainer = DreamBoothTrainer(config, engine, train_dataloader)
    trainer.train()


if __name__ == "__main__":
    main()