import torch
import torch.nn as nn
import yaml

import torch.optim as optim
import torch.nn.functional as F

import logging
from torch.utils.tensorboard import SummaryWriter  # type: ignore

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
    "batch_size": 128,
    "epochs": 40,
    "learning_rate": 0.001,
    "lr_scheduler_step": 10,
    "lr_scheduler_gamma": 0.1,
}
ATTENTION_CONFIG = {"num_heads": 8, "dropout": 0.1, "use_bias": True}
MODEL_CONFIG = {
    "image_size": 224,
    "patch_size": 16,
    "num_classes": 10,  # Number of classes in the dataset
    "hidden_states": 512,
    "num_layers": 12,
    "dropout": 0.1,
}
###### CONFIGURATION ######


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
        "training_config": TRAINING_CONFIG,
    }

    with open(
        os.path.join(EXP_DIR, TRAINING_CONFIG["experiment_name"], "config.yaml"), "w"
    ) as f:
        yaml.dump(config_dict, f)
    logging.info(f"Configuration saved to {os.path.join(model_dir, 'config.yaml')}")

    writer = SummaryWriter(model_dir)

    # Initialize Model
    model = ViT(model_config)
    model.to(device)

    ## Optimizer and Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAINING_CONFIG["lr_scheduler_step"],
        gamma=TRAINING_CONFIG["lr_scheduler_gamma"],
    )
    criterion = nn.CrossEntropyLoss()

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=TRAINING_CONFIG['epochs'],
        device=device,
        model_dir=model_dir,
        scheduler=scheduler,
        writer=writer,
    )
    writer.close()
    logging.info("Training completed.")
    logging.info("TensorBoard logs saved.")
    logging.info("Model saved.")
