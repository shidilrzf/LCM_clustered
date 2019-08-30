import os
from torchvision import transforms, datasets
from random import randint
import torch
import latent_models as cnn



class loader(object):
    def __init__(self,datadir, device, K, model_name=None, img_cluster_id=None ):




        self.end = 0
        self.start = 0
        self.model_name = model_name
        self.device = device
        self.num_clusters = K
        self.num_samples = self.train_dataset.data.shape[0]
        self.temp_latent_dir = 'latent_temp_dir/' + self.model_name + '_latent_temp/'
        if not os.path.exists(self.temp_latent_dir):
            os.makedirs(self.temp_latent_dir)

        if not img_cluster_id:
            self.img_cluster_id = [randint(1, K) for i in range(self.num_samples)]
        else:
            self.img_cluster_id = img_cluster_id

    def load(self, latent_net_name,
                 num_epoch=None,
                 saved_model_name=None,
                 latent_dir=None):


            self.num_epoch = num_epoch
            self.latent_dir = latent_dir
            self.latent_net_name = latent_net_name

            for i in range(self.num_clusters):

                latent_net = getattr(cnn, self.latent_net_name)()
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
            latent_net = getattr(cnn, self.latent_net_name)()
            latent_net = latent_net.to(self.device)
            latent_net.load_state_dict(torch.load(self.temp_latent_dir + "temp_latent_net_" + str(i)))
            latent_nets.append(latent_net)
        return latent_nets

    def save_nets(self, nets, net_ids):
        num_nets = len(net_ids)
        for i in range(num_nets):
            torch.save(nets[i].state_dict(), self.temp_latent_dir + "temp_latent_net_" + str(net_ids[i]))



    def get_batch(self, batch_size=10,img_cluster_id_updated=None):
        if img_cluster_id_updated:
            self.img_cluster_id = img_cluster_id_updated
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
            data_out.append(self.train_dataset.__getitem__(start+i)[0])
            latent_net_ids.append(self.img_cluster_id[start + i])

        latent_nets = self.get_nets(latent_net_ids)
        return data_out, latent_nets, latent_net_ids
















