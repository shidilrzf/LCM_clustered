import os
from random import randint
import torch
import models as lm
from torch.utils.data import Dataset
from skimage.transform import resize
import glob
from skimage.io import imread
import cv2


class CelebAClusterDataset(Dataset):
    def __init__(self, file_dir, net_dir, transform=None):
        self.file_dir = file_dir
        self.net_dir = net_dir
        self.transform = transform
        self.img_list = glob.glob(self.file_dir)
        self.K = 10
        self.img_size = 128
        self.labels = [randint(0, self.K - 1) for _ in range(len(self.img_list))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        cluster_id = self.labels[idx]
        img = imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = resize(img, (self.img_size, self.img_size), mode='constant')
        img = (img - img.min()) / (img.max() - img.min())
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, cluster_id

    def get_assignments(self):
        return self.img_list, self.labels


class loader(object):
    def __init__(self, dataset, device, K, model_name=None):

        self.end = 0
        self.start = 0
        self.model_name = model_name
        self.device = device
        self.num_clusters = K
        self.dataset = dataset
        self.num_samples = len(self.dataset)
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
                    torch.load(latent_dir + saved_model_name + "_latentnet_" + str(num_epoch) + '_' + str(i)))
            torch.save(latent_net.state_dict(), self.temp_latent_dir + "temp_latent_net_" + str(i))
            if num_epoch is None:
                print("Networks loaded: ", i)
        print("Number of samples loaded: ", self.num_clusters)

    def get_nets(self, net_ids):
        latent_nets = []
        for i in net_ids:
            latent_net = getattr(lm, self.latent_net_name)()
            latent_net = latent_net.to(self.device)
            latent_net.load_state_dict(torch.load(self.temp_latent_dir + "temp_latent_net_" + str(i)))
            latent_nets.append(latent_net)
        return latent_nets

    def get_all_nets(self):
        latent_nets = []
        for i in range(self.num_clusters):
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
        img_indices = []

        for i in range(eff_batch):
            img_indices.append(start + i)
            img, cluster_id = self.dataset.__getitem__(start + i)
            data_out.append(img.unsqueeze(0))
            latent_net_ids.append(cluster_id)

        latent_nets = self.get_nets(latent_net_ids)

        # convert lists to batch_num, channel, h, w
        data_out = torch.cat(data_out).type(torch.FloatTensor).to(self.device)

        return data_out, latent_nets, latent_net_ids, img_indices

    def update_cluster_id(self, img_ind, labels):
        for ind, label in zip(img_ind, labels):
            self.dataset.labels[ind] = label

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
        for i in range(self.num_clusters):
            latent_net = self.get_nets([i])[0]
            torch.save(latent_net.state_dict(), latent_dir + name + str(i))

        def get_vec(self, idx):
            return self.data_lst[idx][1]
