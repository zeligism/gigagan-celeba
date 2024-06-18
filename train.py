# Example with 1 x 40G GPUs:
#   python train.py model=gigagan-small batch_size=28 steps=50000 +model.model_folder="gigagan-small-models" +model.results_folder="gigagan-small-results"
# Example with 4 x 40G GPUs:
#   accelerate config   (optional)
#   accelerate launch train.py model=gigagan batch_size=4 steps=50000

import os
import glob
import hydra
from hydra.utils import instantiate
import torch
import torch.distributed
from torchvision import transforms
from torch.utils.data import DataLoader
from celeba import CelebAHQ
from gigagan_pytorch import GigaGAN


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg):
    mps_is_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if mps_is_available else "cpu"))
    print("Device =", device)
    if device.type in ("mps", "cpu"):
        cfg.amp = False  # amp is only used in gpu

    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor()
    ])
    dataset = CelebAHQ(min_occurences=cfg.min_occurences, root=cfg.data_dir, target_type=["attr", "identity"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    cfg.cond_dim = dataset.num_ids + dataset.num_attrs

    gan: GigaGAN = instantiate(cfg.model).to(device)
    # Try loading the latest model
    if cfg.load_latest_model:
        i = -1
        for path in glob.glob(os.path.join(gan.model_folder, f"model-*.ckpt")):
            i = max(i, int(path.split(".ckpt")[-2].split("model-")[-1]))
        if i >= 0:
            gan.load(os.path.join(gan.model_folder, f"model-{i}.ckpt"))
        else:
            print(f"Couldn't find any trained model in '{gan.model_folder}'")
            print("Training model from scratch.")
    # Train
    gan.set_dataloader(dataloader)
    gan(steps=cfg.steps, grad_accum_every=cfg.grad_accum_every)


if __name__ == '__main__':
    main()
