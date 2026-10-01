"""Explicit local-dataset training entry point; importing this module does not start training."""
import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Train a 64px RGB WGAN-GP on a local ImageFolder dataset.")
    p.add_argument("--data-dir", type=Path, help="Directory with images organized into subfolders")
    p.add_argument("--output-dir", type=Path, default=Path("runs/wgan-gp"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--critic-iterations", type=int, default=5)
    p.add_argument("--gradient-weight", type=float, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, help="Stop after this many generator updates")
    p.add_argument("--smoke-test", action="store_true", help="Use synthetic images to check the training path")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.smoke_test and args.data_dir is None:
        parser().error("Specify --data-dir or --smoke-test.")
    if min(args.epochs, args.batch_size, args.critic_iterations) < 1:
        parser().error("Epochs, batch size and critic iterations must be positive.")
    if args.max_steps is not None and args.max_steps < 1:
        parser().error("--max-steps must be positive.")
    import csv
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from .generator import Generator
    from .critic import CriticNetwork
    from .gradient_penalty import gradient_penalty

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                    transforms.Normalize([0.5] * 3, [0.5] * 3)])
    if args.smoke_test:
        dataset = datasets.FakeData(size=max(args.batch_size, 8), image_size=(3, 64, 64), transform=transform)
    else:
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    generator = Generator(100, 3, 64).to(device)
    critic = CriticNetwork(3, 64).to(device)
    for network in (generator, critic):
        for module in network.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1, 0.02)
                nn.init.zeros_(module.bias)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.learning_rate, betas=(0, 0.9))
    fixed_noise = torch.randn(32, 100, 1, 1, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        log = csv.writer(stream)
        log.writerow(["step", "epoch", "critic_loss", "generator_loss", "gradient_penalty"])
        for epoch in range(args.epochs):
            for real, _ in loader:
                real = real.to(device)
                for _ in range(args.critic_iterations):
                    with torch.no_grad():
                        fake = generator(torch.randn(real.shape[0], 100, 1, 1, device=device))
                    penalty = gradient_penalty(critic, real, fake)
                    loss_c = critic(fake).mean() - critic(real).mean() + args.gradient_weight * penalty
                    opt_c.zero_grad(set_to_none=True)
                    loss_c.backward()
                    opt_c.step()
                for param in critic.parameters():
                    param.requires_grad_(False)
                fake = generator(torch.randn(real.shape[0], 100, 1, 1, device=device))
                loss_g = -critic(fake).mean()
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
                for param in critic.parameters():
                    param.requires_grad_(True)
                step += 1
                log.writerow([step, epoch, loss_c.item(), loss_g.item(), penalty.item()])
                stream.flush()
                if step % 100 == 0 or step == 1:
                    print(f"step={step} critic={loss_c.item():.4f} generator={loss_g.item():.4f}")
                    generator.eval()
                    with torch.no_grad():
                        save_image(generator(fixed_noise), args.output_dir / f"samples-{step:06d}.png", normalize=True, value_range=(-1, 1))
                    generator.train()
                if args.max_steps is not None and step >= args.max_steps:
                    break
            if args.max_steps is not None and step >= args.max_steps:
                break
    torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                "optimizer_generator": opt_g.state_dict(), "optimizer_critic": opt_c.state_dict(),
                "step": step, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
               args.output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
