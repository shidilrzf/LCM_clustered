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

# Cluster winner takes all
parser.add_argument('--wta', action='store_true', default=False,
                    help='Winner takes all, only update best subnet (default: True')


args = parser.parse_args()

# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

start_epoch, end_epoch = 0, args.epochs

# Logging to default runs/CURRENT_DATETIME_HOSTNAME
writer = SummaryWriter()

# models
model_name = 'CelebA_cluster'
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

# Get initial cluster assignments for visualization
init_img_list, init_labels = train_dataset.get_assignments()

cluster_grid = visualize_clusters(init_img_list, init_labels)

writer.add_image('clusters', cluster_grid, -1)


# Init and load latent space networks
dataloader.load(latent_net_name, None, None, latentnet_fp)

# Generate uniform noise
noise = sample_uniform((1, 2, noise_sz, noise_sz), -1, 1, 1. / 10, args.cuda)
noise = noise.to(device)

# Optimizer
gen_optimizer = optim.SGD(generator.parameters(), lr=0.03)


# main train_validate loop
def train(epoch, data_in, latent_net_ids, net_in, num_epochs=args.epochs, use_cuda=False):
    """
     Jointly trains the latent networks and the generator network.
     """
    data_in = data_in.cuda() if use_cuda else data_in

    assert(data_in.size(0) == len(latent_net_ids))

    generator.train()
    for p in generator.parameters():
        p.requires_grad = True

    # Might be wrong
    batch_size = len(data_in)
    nets_params = []

    for i in range(batch_size):
        ind = latent_net_ids[i]
        net_in[ind] = net_in[ind].cuda() if use_cuda else net_in[ind]
        for p in net_in[ind].parameters():
            p.requires_grad = True
        nets_params += list(net_in[ind].parameters())

    latent_optimizer = optim.SGD(nets_params, lr=0.03, weight_decay=0.001)

    for ep in range(num_epochs):
        gen_optimizer.zero_grad()
        latent_optimizer.zero_grad()

        map_out_lst = []
        for i in range(batch_size):
            ind = latent_net_ids[i]
            m_out = net_in[ind](noise)
            map_out_lst.append(m_out)
        map_out = torch.cat(map_out_lst, 0)
        g_out = generator(map_out)

        lap_loss = laploss(g_out, data_in)
        mse_loss = F.mse_loss(g_out, data_in)
        loss = mse_loss + lap_loss
        loss.backward()

        latent_optimizer.step()
        gen_optimizer.step()
        if args.clamp:
            for i in range(batch_size):
                ind = latent_net_ids[i]
                net_in[ind].restrict(-1.0 * args.clamp_value, args.clamp_value)

    latent_optimizer.zero_grad()
    gen_optimizer.zero_grad()

    # Logging
    print('====> Epoch: {} Average Train loss: {:.4f}'.format(epoch, loss.item()))

    if epoch % 10 == 0:
        generator.eval()
        map_out_lst = []
        for i in range(batch_size):
            ind = latent_net_ids[i]
            m_out = net_in[ind](noise)
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


def update_cluster(data_in, net_in, img_indices, use_cuda):

    data_in = data_in.cuda() if use_cuda else data_in
    generator.train()

    batch_size = len(data_in)
    num_cluster = len(net_in)
    labels = []

    for i in range(batch_size):
        ind = latent_net_ids[i]
        net_in[ind] = net_in[ind].cuda() if use_cuda else net_in[ind]

    for i in range(batch_size):
        scores = []
        for ind in range(num_cluster):
            map_out = net_in[ind](noise)
            g_out = generator(map_out)

            lap_loss = laploss(g_out, data_in[i].unsqueeze(0))
            mse_loss = F.mse_loss(g_out, data_in[i].unsqueeze(0))
            loss = mse_loss + lap_loss
            scores.append(loss.item())

        # find lowest score
        min_ind = np.argmin(scores)
        labels.append(min_ind)

    return labels


for epoch in range(start_epoch, end_epoch + 1):
    #Get a batch of images, their latent networks and corresponding network ids
    data_in, latent_nets, latent_net_ids, img_indices = dataloader.get_batch(batch_size=args.batch_size)

    #train the latent networks and generator
    latent_nets = train(epoch, data_in, latent_net_ids, latent_nets, num_epochs=50, use_cuda=args.cuda)
    dataloader.update_state(latent_nets, latent_net_ids)

    if args.wta:
        # Update cluster
        latent_nets = dataloader.get_all_nets()
        new_labels = update_cluster(data_in, latent_nets, img_indices, use_cuda=args.cuda)
        dataloader.update_cluster_id(img_indices, new_labels)
        # Display cluster
        img_list = [dataloader.dataset.img_list[ind] for ind in img_indices]
        print(img_list)
        print(new_labels)

        assert(len(img_list) == len(new_labels))
        cluster_grid = visualize_clusters(img_list, new_labels)
        writer.add_image('clusters', cluster_grid, epoch)

    if epoch % 100 == 0:
        if epoch > 0:
            dataloader.save_latent_net(name=model_name + "_latentnet_" + str(epoch) + "_", latent_dir=latentnet_fp)
            torch.save(generator.state_dict(), 'models/CelebA/' + model_name + str(epoch))

writer.close()
