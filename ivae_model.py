import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
import tqdm
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['CUDA_VISIBLE_DEVICES']='6,7'
device='cuda'
download_root='./data'

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] ='localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



def prepare(rank, world_size, batch_size=64, pin_memory=False, num_workers=8):
    dataset = MNIST(download_root,train=True,transform=transforms.ToTensor(),download=False)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader


class ConvVae(nn.Module):
    '''
       Identifiable VAE:

        Image data $\mathbf{x}$, auxiliary variable $\mathbf{y}$ (label), latent variable $\mathbf{z}$ 


    '''
    def __init__(self,z_dim):
        super().__init__()

        self.z_dim=z_dim

        ## Encoder
        self.encoder=nn.Sequential(
            ## Input size: C x H x W = 1 x 28 x 28

            nn.Conv2d(1,32,4,2,1),
            ## C x H x W = 32 x 14 x 14
            nn.GroupNorm(4,32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,64,4,2,1),
            ## C x H x W = 64 x 7 x 7
            nn.GroupNorm(8,64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,7,1,0),
            ## C x H x W = 128 x 1 x 1
        )
        ## Gaussian posterior distribution $q_\phi(\mathbf{z|x,y})$ which approximates to $p_\theta(\mathbf{z|x,y})$
        self.fc_mu= nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,z_dim)
        )
        self.fc_std=nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,z_dim)
        )
        ## Decoder
        self.decoder= nn.Sequential(
            ## Input size: C x H x W = z_dim x 1 x 1

            nn.ConvTranspose2d(z_dim, 128, 1,1),
            ## C x H x W = 128 x 1 x 1
            nn.GroupNorm(16,128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128,64,7,1),
            ## C x H x W = 64 x 7 x 7 
            nn.GroupNorm( 8,64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64,32,4,2,1),
            ## C x H x W = 32 x 14 x 14
            nn.GroupNorm(4,32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32,1,4,2,1),
            ## C x H x W = 1 x 28 x 28
        )

        ## Concatenate Y to X
        self.Emb = nn.Linear(28*28+10,28*28)

        ## Prior distribution with auxiliary variable $\mathbf{y}$: $p_\lambda(\mathbf{z|y})$ selected as Gaussian
        self.p = nn.Sequential(
                            nn.Linear(10,64),
                            nn.ReLU(inplace=True),
                            nn.Linear(64,128))
        self.p_mu = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,z_dim)
        )
        self.p_std = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,z_dim)
        )


    def encode(self, x, y):
        ## Make a distribution $p_\phi(\mathbf{z|x,y})$
        ## return: mean and std of the distribution
        
        ## Input: B x H x C x W = B x 1 x 28 x 28 

        h = torch.cat((x.view(-1,1,28*28),y.view(-1,1,10)),dim=-1)
        h = self.Emb(h)
        h = self.encoder(h.view(-1,1,28,28))
        ## h: C x H x W = 128 x 1 x 1
        ## To be an input of linear layer, squeeze it.
        h=h.squeeze()
        return self.fc_mu(F.relu(h)) , self.fc_std(F.relu(h))

    def decode(self, z):
        ## Make a distribution $p_\theta(\mathbf{x|z,y})$
        ## return: generated image
        
        h = self.decoder(z.view(-1,self.z_dim,1,1))
        ## h: C x H x W = 1 x 28 x 28

        ## Make each pixel's value to be in [0,1]
        return torch.sigmoid(h.view(-1,28*28)) 

    def prior(self, y):
        ## Make a distribution $p_\lambda(\mathbf{z|y})$
        ## return: mean and std of the distribution
        h = F.relu(self.p(y))
        return self.p_mu(h), self.p_std(h)

    @staticmethod
    def reparam(mu,logv):
        noise = torch.randn_like(logv)
        v = torch.exp(logv*.5) 
        return mu+ noise*v
    
    def forward(self,x,y):
        ## Encoder: $q_\phi(\mathbf{z|x,y})$
        mu, logv = self.encode(x,y)
            
         ## Prior: $p_{T,\lambda}(\mathbf{z|y})$ 
        p_mu,p_logv = self.prior(y)
            
        ## Decoder: $p_f(\mathbf{x|z})$
        z = self.reparam(mu,logv)
        x_recon = self.decode(z)
        
        return x_recon, p_logv, p_mu, mu, logv


def ivaeloss(x, x_recon, p_logv, p_mu, mu, logv):
    ## ELBO: \mathcal{E}_{q_phi(\mathbf{z|x,y})}[\log p_\theta(\mathbf{x,z|y})]-\log q_\phi(\mathbf{z|x,y})]
    ##     = \mathcal{E}_{q_phi(\mathbf{z|x,y})}[\log p_\theta(\mathbf{x|z})+\log p_\theta(\mathbf{z|y}) - \log q_\phi(\mathbf{z|x,y})]
    
    ## recon_loss: \mathcal{E}_{q_phi(\mathbf{z|x,u})}[\log p_\theta(\mathbf{x|z})}
    recon_loss= F.mse_loss(x_recon, x.view(-1, 28*28),reduction='sum')
    
    ## neg-kl_divergence: \mathcal{E}_{q_phi(\mathbf{z|x,y})}[\log p_\theta(\mathbf{z|y}) - \log q_\phi(\mathbf{z|x,y})]
    ##              = \frac{1}{2}\sum_{j=1}^{J}{\big(1+\log{\sigma^2_{encoder}}-\log{\sigma^2_{prior}}-\frac{\sigma^2_{encoder}}{\sigma^2_{prior}}-\frac{(\mu_{prior}-\mu_{encoder})^2}{\sigma^2_{prior}}\big)}
    ## kl_loss = kl_divergence
    
    kl_loss= 0.5*torch.sum(-1 - logv +(logv.exp()+(p_mu-mu).pow(2))/(p_logv.exp()+1e-08)+p_logv)
    
    return recon_loss , kl_loss

def one_hot(labels,class_size,rank):
    targets = torch.zeros(labels.size(0),class_size).to(rank)
    for i , label in enumerate(labels):
        targets[i, label]=1
    return targets 
  


lr= 1e-04
epochs=50

def cleanup():
    dist.destroy_process_group()


def main(rank,world_size):
    setup(rank,world_size)

    dataloader=prepare(rank,world_size)

    model = ConvVae(z_dim=2).to(rank)
    model= DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=True)

    optimizer= Adam(model.parameters(),lr=lr)
    tqdm_epoch=tqdm.trange(epochs)
    for epoch in tqdm_epoch:
        dataloader.sampler.set_epoch(epoch)
        avg_loss=0.
        num_items=0
        for x,y in dataloader:
            x=x.to(rank)
            y=one_hot(y,10,rank)

            x_recon, p_logv, p_mu, mu, logv=model(x,y)
            recon_loss, kl_loss= ivaeloss(x,x_recon,p_logv,p_mu, mu,logv)
            loss=recon_loss+kl_loss
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            avg_loss+=loss.item() * x.shape[0]
            
            num_items+=x.shape[0]
        tqdm_epoch.set_description('Average loss: {:5f}'.format(avg_loss/num_items))
        ##cleanup()
    torch.save(model.module.state_dict(),'/DataCommon2/wtjeong/model/VAE2.pt')

import torch.multiprocessing as mp
if __name__ == '__main__':
    
    world_size = 2    
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
    
