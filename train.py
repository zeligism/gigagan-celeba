from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import PseudoTextEncoder
from utils import CelebAHQ


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    mps_is_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if mps_is_available else "cpu"))
    print("Device =", device)
    if device.type in ("mps", "cpu"):
        cfg.amp = False

    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor()
    ])
    dataset = CelebAHQ(min_occurences=cfg.min_occurences, root=cfg.data_dir, target_type=["attr", "identity"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    cfg.model.generator.text_encoder.dim_in = dataset.num_ids + dataset.num_attrs
    cfg.model.discriminator.text_encoder.dim_in = dataset.num_ids + dataset.num_attrs

    gan = instantiate(cfg.model).to(device)
    gan.set_dataloader(dataloader)

    # Train
    gan(steps=cfg.steps, grad_accum_every=cfg.grad_accum_every)

    # after much training
    images = gan.generate(batch_size=4)


if __name__ == '__main__':
    main()
