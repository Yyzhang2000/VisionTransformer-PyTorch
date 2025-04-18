import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

import logging

import shutil


def train_one_epoch(
    model, train_loader, optimizer, criterion, scheduler, device, epoch, writer=None
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"
    )
    for batch_idx, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        if writer:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        if writer and epoch is not None:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", 100 * correct / total, step)

        del images, labels, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def test(model, test_loader, criterion, device, writer=None, epoch=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Test Epoch {epoch+1}" if epoch is not None else "Evaluating",
        )
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
            if writer and epoch is not None:
                step = epoch * len(test_loader) + batch_idx
                writer.add_scalar("Loss/test", loss.item(), step)
                writer.add_scalar("Accuracy/test", 100 * correct / total, step)

            del images, labels, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    num_epochs=20,
    scheduler=None,
    model_dir: str = "./logs/model",
    writer=None,
):
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, epoch, writer
        )
        test_loss, test_acc = test(model, test_loader, criterion, device, writer, epoch)

        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

        torch.save(
            model.state_dict(),
            os.path.join(model_dir, f"latest_model.pth"),
        )
        if test_acc > best_acc:
            best_acc = test_acc
            shutil.copyfile(
                os.path.join(model_dir, "latest_model.pth"),
                os.path.join(model_dir, "best_model.pth"),
            )

            logging.info(f"Best model saved with accuracy: {best_acc:.2f}%")

    logging.info(f"Training complete. Best test accuracy: {best_acc:.2f}%")
