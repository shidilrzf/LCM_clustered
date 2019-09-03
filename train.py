import torch
import torch.optim
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as tvu
from data import *
from models import *
from losses import *
from utils import *
import os
import argparse


parser = argparse.ArgumentParser(description='Cluster LCM')

# Settings cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

# Training
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of training epochs')

parser.add_argument('--num-clusters', type=int, default=10, metavar='N',
                    help='number of latent clusters (default: 10)')

parser.add_argument('--clamp', action='store_true', default=True,
                    help='Clamps values of training images (default: True')

parser.add_argument('--clamp-value', type=float, default=0.01,
                    help='Clamp values (default: 0.01')

# Optim
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='Batch size (default: 10)')


args = parser.parse_args()

# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

start_epoch, end_epoch = 0, args.epochs

# Logging to default runs/CURRENT_DATETIME_HOSTNAME
writer = SummaryWriter()

# models
model_name = 'CelebA_cluster1'
latent_net_name = 'Latent4LSND'
latentnet_fp = '../' + model_name + '_latent/'

if not os.path.exists(latentnet_fp):
    os.makedirs(latentnet_fp)

# Generator model
generator = EncDecCelebA(in_channels=64)
generator.to(device)


# Datasets and loaders
noise_sz = 40
dataset_fp = '/data/CelebA/celebA_redux_500/*.png'

# Build dataset
train_dataset = CelebAClusterDataset(dataset_fp, latentnet_fp, None)
dataloader = loader(train_dataset, device, args.num_clusters, model_name)

# Init and load latent space networks
dataloader.load(latent_net_name, None, None, latentnet_fp)

# Generate uniform noise
noise = sample_uniform((1, 2, noise_sz, noise_sz), -1, 1, 1. / 10, args.cuda)
noise = noise.to(device)

# Optimizer
gen_optimizer = optim.SGD(generator.parameters(), lr=0.03)


# main train_validate loop
def train(epoch, data_in, net_in, num_epochs=args.epochs, use_cuda=False):
    """
     Jointly trains the latent networks and the generator network.
     """
    data_in = data_in.cuda() if use_cuda else data_in

    generator.train()
    for p in generator.parameters():
        p.requires_grad = True

    batch_size = len(net_in)
    nets_params = []

    for i in range(batch_size):
        net_in[i] = net_in[i].cuda() if use_cuda else net_in[i]
        for p in net_in[i].parameters():
            p.requires_grad = True
        nets_params += list(net_in[i].parameters())

    optim_nets = optim.SGD(nets_params, lr=0.03, weight_decay=0.001)

    for ep in range(num_epochs):
        gen_optimizer.zero_grad()
        optim_nets.zero_grad()

        map_out_lst = []
        for i in range(batch_size):
            m_out = net_in[i](noise)
            map_out_lst.append(m_out)
        map_out = torch.cat(map_out_lst, 0)
        g_out = generator(map_out)

        lap_loss = laploss(g_out, data_in)
        mse_loss = F.mse_loss(g_out, data_in)
        loss = mse_loss + lap_loss
        loss.backward()

        optim_nets.step()
        gen_optimizer.step()
        if args.clamp:
            for i in range(batch_size):
                net_in[i].restrict(-1.0 * args.clamp_value, args.clamp_value)
    optim_nets.zero_grad()
    gen_optimizer.zero_grad()

    # Logging
    print('====> Epoch: {} Average Train loss: {:.4f}'.format(epoch, loss.item()))

    if epoch % 10 == 0:
        generator.eval()
        map_out_lst = []
        for i in range(batch_size):
            m_out = net_in[i](noise)
            map_out_lst.append(m_out)

        map_out = torch.cat(map_out_lst, 0)
        g_out = generator(map_out)

        writer.add_scalar('loss', loss.data.item(), epoch)
        writer.add_scalar('laplacian', lap_loss.item(), epoch)
        writer.add_scalar('MSE', mse_loss.item(), epoch)

        image_grid = tvu.make_grid(data_in[:5].cpu().detach(), normalize=False, scale_each=True)
        gen_grid = tvu.make_grid(g_out[:5].cpu().detach(), normalize=False, scale_each=True)
        latent_grid = tvu.make_grid(map_out[:10, :3, :, :].cpu().detach(), normalize=False, scale_each=True)

        writer.add_image('real', image_grid, epoch)
        writer.add_image('generated', gen_grid, epoch)
        writer.add_image('latent', latent_grid, epoch)

    return net_in


for epoch in range(start_epoch, end_epoch + 1):
    #Get a batch of images, their latent networks and corresponding network ids
    data_in, latent_nets, latent_net_ids = dataloader.get_batch(batch_size=args.batch_size)

    #train the latent networks and generator
    latent_nets = train(epoch, data_in, latent_nets, num_epochs=50, use_cuda=args.cuda)

    #update the latent networks
    dataloader.update_state(latent_nets, latent_net_ids)
    if epoch % 100 == 0:
        if epoch > 0:
            dataloader.save_latent_net(name=model_name + "_latentnet_" + str(epoch) + "_", latent_dir=latentnet_fp)
            torch.save(generator.state_dict(), '../models/CelebA/' + model_name + str(epoch))

writer.close()
