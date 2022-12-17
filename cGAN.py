import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty
from model import Critic, Generator, initialize_weights
from tqdm import tqdm

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-4
batch_size = 100
img_size = 28
img_channels = 1
z_dim = 100
epochs = 5
features_critic = 16
features_gen = 16
critic_iterations = 5
lambda_gp = 10
num_classes = 10
gen_embedding = 100

transforms = transforms.Compose(
    [
        #transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)]),
    ]
)

dataset = datasets.MNIST(root="/files/", train=True, download=True, transform=transforms)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(z_dim, img_channels, features_gen, num_classes, img_size, gen_embedding).to(device)
critic = Critic(img_channels, features_critic, num_classes, img_size).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter(f"runs/real")
writer_fake = SummaryWriter(f"runs/fake")
step = 1

gen.train()
critic.train()

for epoch in range(epochs):
    lossC_pe = 0
    lossG_pe = 0
    train_loader = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch: {epoch + 1}")
    for batch_idx, (real, labels) in train_loader:
        real = real.to(device)
        labels = labels.to(device)
        BS = real.shape[0]
        fr = (batch_size / BS) * len(loader)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for critic_iteration in range(critic_iterations):
            noise = torch.randn(BS, z_dim, 1, 1).to(device)
            fake = gen(noise, labels)

            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)

            gp = gradient_penalty(critic, labels, real, fake, device=device)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            if critic_iteration == critic_iterations - 1:
                with torch.no_grad():
                    lossC_pe += loss_critic.item()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        output = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        with torch.no_grad():
            lossG_pe += loss_gen.item()

        # Print losses occasionally and print to tensorboard

        if (batch_idx % 100) == 0:
            with torch.no_grad():
                fake = gen(noise, labels)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

        train_loader.set_postfix(loss_critic=fr * loss_critic.item(),
                                 loss_gen=fr * loss_gen.item())

    print(f"Epoch: {epoch+1}, Loss critic: {lossC_pe}, Loss genenerator: {lossG_pe}")
    writer_real.add_scalar("Loss critic", lossC_pe, global_step=epoch+1)
    writer_fake.add_scalar("Loss generator", lossG_pe, global_step=epoch+1)

torch.save(critic, "./critic.pth")
torch.save(gen, "./generator.pth")

print("Models are saved")