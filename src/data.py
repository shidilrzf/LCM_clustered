import os
from random import randint
import torch
import torch.utils.data.Dataset as Dataset
import models as lm
import glob
from skimage.io import imread


class CelebAClusterDataset(Dataset):
    def __init__(self, file_dir, net_dir, transform=None):
        self.file_dir = file_dir
        self.net_dir = net_dir
        self.transform = transform
        self.img_list = [file_dir + '/' + f for f in glob.glob('*.png')]
        self.K = 10

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        cluster_id = randint(0, self.K - 1)
        img = torch.from_numpy(imread(self.img_list[idx])).permute(2, 0, 1)
        return img, cluster_id


class loader(object):
    def __init__(self, dataset, device, K, model_name=None):

        self.end = 0
        self.start = 0
        self.model_name = model_name
        self.device = device
        self.num_clusters = K
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.temp_latent_dir = 'latent_temp_dir/' + self.model_name + '_latent_temp/'

        if not os.path.exists(self.temp_latent_dir):
            os.makedirs(self.temp_latent_dir)

    def load(self, latent_net_name,
             num_epoch=None,
             saved_model_name=None,
             latent_dir=None):

        self.num_epoch = num_epoch
        self.latent_dir = latent_dir
        self.latent_net_name = latent_net_name

        for i in range(self.num_clusters):

            latent_net = getattr(lm, self.latent_net_name)()
            latent_net = latent_net.to(self.device)
            if num_epoch:
                if saved_model_name is None:
                    raise ValueError("\'saved_model_name\' must be specified when loading a saved model")
                if latent_dir is None:
                    raise ValueError("\'latent_dir\' must be specified when loading a saved model")
                print("Num prev Loaded: ", self.num_samples)
                latent_net.load_state_dict(
                    torch.load(latent_dir + saved_model_name + "_latentnet_" + str(num_epoch) + '_' + str(i + 1)))
            torch.save(latent_net.state_dict(), self.temp_latent_dir + "temp_latent_net_" + str(i + 1))
            if num_epoch is None:
                print("Networks loaded: ", i + 1)
        print("Number of samples loaded: ", self.num_clusters)

    def get_nets(self, net_ids):
        latent_nets = []
        for i in net_ids:
            latent_net = getattr(lm, self.latent_net_name)()
            latent_net = latent_net.to(self.device)
            latent_net.load_state_dict(torch.load(self.temp_latent_dir + "temp_latent_net_" + str(i)))
            latent_nets.append(latent_net)
        return latent_nets

    def save_nets(self, nets, net_ids):
        num_nets = len(net_ids)
        for i in range(num_nets):
            torch.save(nets[i].state_dict(), self.temp_latent_dir + "temp_latent_net_" + str(net_ids[i]))

    def get_batch(self, batch_size=10):
        self.start = self.end
        start = self.start
        end = min(start + batch_size, self.num_samples)
        eff_batch = end - start
        if end >= self.num_samples:
            end = 0
        self.end = end
        data_out = []
        latent_net_ids = []

        for i in range(eff_batch):
            img, cluster_id = self.dataset.__getitem__(start + i)
            data_out.append(img)
            latent_net_ids.append(cluster_id)

        latent_nets = self.get_nets(latent_net_ids)
        return data_out, latent_nets, latent_net_ids

    def update_state(self, latent_nets, latent_nets_ids):
        """
          Updates the latent networks during training.
          Parameters:
                    latent_nets (List): List of latent networks (nn.Modules) to be saved.
                    latent_nets_ids (List): List of ids of the latent networks to be saved.
          """
        self.save_nets(latent_nets, latent_nets_ids)

    def save_latent_net(self, latent_dir, name):
        """
          Saves the latent networks in `latent_dir` to create a model checkpoint.
          Parameters:
                    latent_dir (String): Name of directory where the latent networks are saved.
                    name (String): Name with which the latent networks must be saved.
          """
        if not os.path.exists(latent_dir):
            os.makedirs(latent_dir)
        for i in range(self.K):
            latent_net = self.get_nets([i])[0]
            torch.save(latent_net.state_dict(), latent_dir + name + str(i))

        def get_vec(self, idx):
            return self.data_lst[idx][1]
