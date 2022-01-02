from option import args
from model import SRCNN,Unet
from dataset import *
from torch.utils.data import DataLoader
import torch
import torch.utils
from torch.nn import MSELoss
from tqdm import tqdm
from utils import SRUtils
import torchvision.models as models
from time import time
from datetime import datetime
# -*- coding: UTF-8 -*-
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

class Train():
    def __init__(self,args):
        #---train dataset init---#
        self.dataloader=DataLoader(AstroData(args),batch_size=args.batch_size)

        #--- model and training init ---#
        self.time=int(time())
        # self.net=SRCNN(args)
        if not os.path.exists('net.pkl'):
            self.net=Unet(args)
        else:
            self.net=torch.load('net.pkl')
            print('Load Model Success')

        self.optimizer=torch.optim.Adam(lr=args.lr_rate,params=self.net.parameters())
        self.loss_fuc=MSELoss().to(device)
        self.net=self.net.to(device)
        self.losses=[]
        self.epochs=args.epochs
        self.SRUtils=SRUtils()


    
    def train(self):
        print("Training Start:")

        plot_losses=[]
        for epoch in tqdm(range(int(self.epochs))):
            self.epoch_now=epoch
            for index,data in enumerate(self.dataloader):
                hr_img,lr_img=data
                hr_img,lr_img=self.prepare(hr_img,lr_img)
                sr_img=self.net(lr_img)
                self.loss=self.loss_fuc(sr_img,hr_img)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self.losses.append(self.loss.cpu().detach())

            self.visualize(sr_img.cpu().detach(),hr_img.cpu().detach(),lr_img.cpu().detach())
            loss_cpu=sum(self.losses).item()/len(self.losses)
            plot_losses.append(loss_cpu)
            evl_val=self.SRUtils.RSNR(sr_img.cpu().detach(),hr_img.cpu().detach())
            self.log(loss_cpu,evl_val)
            self.losses=[]
            plt.figure()
            plt.plot(plot_losses)
            plt.savefig(f'loss_curve_{self.time}.png')
            
            if loss_cpu==min(plot_losses):
                torch.save(self.net,'net.pkl')
                

    def visualize(self,sr_img,hr_img,lr_img):
        sr_img=sr_img[0].view(512,512,3).numpy()
        hr_img=hr_img[0].view(512,512,3).numpy()
        lr_img=lr_img[0].view(512,512,3).numpy()

        plt.subplot(2,2,1)
        plt.title('lr_img')
        plt.imshow(lr_img)
        plt.subplot(2,2,2)
        plt.title('SR_img')
        plt.imshow(sr_img)
        plt.subplot(2,2,(3,4))
        plt.title('HR_img')
        plt.imshow(hr_img)
        plt.savefig(f'output_{self.time}.png')

    def log(self,loss_cpu,evl_val):
        with open(f'log_{self.time}.txt','a') as f:
            f.write(f'epoch :{self.epoch_now}, with loss:{loss_cpu}, RSNR:{evl_val}, time:{datetime.now().strftime("%m/%d_%H:%M:%S")}\n')

    def prepare(self, *args):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if len(args) >= 1:
            return (a.float().to(device) for a in args)
        