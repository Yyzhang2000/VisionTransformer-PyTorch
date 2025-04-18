import torch
import torch.nn as nn
import yaml

import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

import logging

from model.vit import ViT
from model.config import ViTConfig, AttentionConfig

from torch.utils.data import DataLoader
from dataset import FruitDataset


from utils import *
from train_engine import train


###### CONFIGURATION ######
DATA_DIR = "./data/MY_data"
EXP_DIR = "./logs"
TRAINING_CONFIG = {
    "experiment_name": "vit_fruit_classification",
    "seed": 42,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 0.001,
    "betas": (0.9, 0.999),
    "weight_decay": 0.01,
}
ATTENTION_CONFIG = {"num_heads": 8, "dropout": 0.1, "use_bias": True}
MODEL_CONFIG = {
    "image_size": 224,
    "patch_size": 8,
    "num_classes": 10,  # Number of classes in the dataset
    "hidden_states": 512,
    "num_layers": 8,
    "dropout": 0.1,
}
###### CONFIGURATION ######


#### Optimization
def warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":
    model_dir = os.path.join(EXP_DIR, TRAINING_CONFIG["experiment_name"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    set_seed(TRAINING_CONFIG["seed"])

    set_logger(os.path.join(model_dir, "train.log"))
    logging.getLogger()

    # device =
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    logging.info(f"Using device: {device}")

    ### Load dataset
    train_dataset = FruitDataset(DATA_DIR, split="train")
    test_dataset = FruitDataset(DATA_DIR, split="test")
    train_loader = DataLoader(
        train_dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=False
    )

    ### Load and Save Config
    MODEL_CONFIG.update(num_classes=len(train_dataset.classes))
    custom_attention = AttentionConfig(**ATTENTION_CONFIG)
    model_config = ViTConfig(**MODEL_CONFIG)
    model_config.attention_config = custom_attention
    config_dict = {
        "model_config": config_to_dict(model_config),
        "training_config": config_to_dict(TRAINING_CONFIG),
    }

    with open(
        os.path.join(EXP_DIR, TRAINING_CONFIG["experiment_name"], "config.yaml"), "w"
    ) as f:
        yaml.dump(config_dict, f)
    logging.info(f"Configuration saved to {os.path.join(model_dir, 'config.yaml')}")
    breakpoint()

    writer = set_tensorboard_writer(model_dir)

    # Initialize Model
    model = ViT(model_config)
    model.apply(initialize_weights)
    logging.info("Model weights initialized.")

    model.to(device)

    ## Optimizer and Learning Rate Scheduler
    total_steps = len(train_loader) * TRAINING_CONFIG["epochs"]
    warmup_steps = int(0.1 * total_steps)

    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        betas=TRAINING_CONFIG["betas"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )
    scheduler = warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss()

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=TRAINING_CONFIG["epochs"],
        device=device,
        model_dir=model_dir,
        scheduler=scheduler,
        writer=writer,
    )
    writer.close()
    logging.info("Training completed.")
    logging.info("TensorBoard logs saved.")
    logging.info("Model saved.")
