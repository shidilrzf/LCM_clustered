from tensorboardX import SummaryWriter
import torch
import torch.optim
import torch.nn.functional as F
import torch.optim as optim
from data import *
from models import *
from losses import *
from utils import *
import os


# Settings
use_cuda = False
use_cuda = use_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

clamp_images = True
num_cluster = 10
start_epoch, end_epoch = 0, 100

writer = SummaryWriter(log_dir='../runs')

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
batch_size = 10
num_train = 500
noise_sz = 40
dataset_fp = '/data/CelebA/celebA_redux_500/*.png'

# Build dataset
train_dataset = CelebAClusterDataset(dataset_fp, latentnet_fp, None)
dataloader = loader(train_dataset, device, num_cluster, model_name)

# Init and load latent space networks
dataloader.load(latent_net_name, None, None, latentnet_fp)

# Generate uniform noise
noise = sample_uniform((1, 2, noise_sz, noise_sz), -1, 1, 1. / 10, use_cuda)
noise = noise.to(device)

# Optimizer
gen_optimizer = optim.SGD(generator.parameters(), lr=0.03)

# main train_validate loop


def train(epoch, data_in, net_in, num_epochs=100):
    """
     Jointly trains the latent networks and the generator network.
     """
    generator.train()
    for p in generator.parameters():
        p.requires_grad = True

    batch_size = len(net_in)
    nets_params = []

    for i in range(batch_size):
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
        if RESTRICT:
            val = RESTRICT_VAL
            for i in range(batch_size):
                net_in[i].restrict(-1.0 * val, val)
    optim_nets.zero_grad()
    gen_optimizer.zero_grad()
    # tensorboard reporting here
    return net_in


for epoch in range(start_epoch, end_epoch + 1):
    #Get a batch of images, their latent networks and corresponding network ids
    data_in, latent_nets, latent_net_ids = dataloader.get_batch(batch_size=batch_size)

    #train the latent networks and generator
    latent_nets = train(epoch, data_in, latent_nets, num_epochs=50)

    #update the latent networks
    dataloader.update_state(latent_nets, latent_net_ids)
    print(fname + " Epoch: ", epoch)
    if epoch % SAVE_EVERY == 0:
        if epoch > 0:
            dataloader.save_latent_net(name=MODEL_NAME + "_latentnet_" + str(epoch) + "_", latent_dir=LATENT_CHECK_DIR)
            torch.save(G.state_dict(), '../models/CelebA/' + MODEL_NAME + str(epoch))

# writer.close()
