from torch import nn
import torch
from option import args
import math

class Unet(nn.Module):
    def __init__(self,args):
        super(Unet,self).__init__()
        #---config init---#
        self.num_channels=args.num_init_layers_channel*3
        self.depth=args.depth
        self.channels_num=[self.num_channels*(2**i) for i in range(self.depth)]#[64,128,256,512,1024]

        #---model block init---#
        self.MaxPooling_Layer=nn.MaxPool2d(2)
        self.finallayer_decode=nn.Conv2d(self.num_channels,3,kernel_size=1)
        
        self.Upconv_Layers=[
            nn.Sequential(nn.ConvTranspose2d(self.channels_num[depth+1],self.channels_num[depth],kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(self.channels_num[depth]),
            nn.ReLU()
            ) for depth in range(self.depth-2,-1,-1)#0:3,2,1,0    +1:4,3,2,1
        ]

        # self.Upconv_Layers=[
        #     nn.Sequential(Upsampler(2,self.channels_num[depth+1],act=False),
        #     # nn.BatchNorm2d(self.channels_num[depth]),
        #     nn.ReLU(),
        #     nn.Conv2d(self.channels_num[depth+1],self.channels_num[depth],kernel_size=3,stride=1,padding=1),
        #     nn.ReLU()
        #     ) for depth in range(self.depth-2,-1,-1)#0:3,2,1,0 
        # ]

        self.softmax=nn.Softmax()
        #---first layer encode init---#
        self.firstlayer_encode=[]
        self.firstlayer_encode.extend([nn.Conv2d(3,self.num_channels,kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(self.num_channels),
                    nn.ReLU()])
        for i in range(2):
            self.firstlayer_encode.extend([nn.Conv2d(self.num_channels,self.num_channels,kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(self.num_channels),
                    nn.ReLU()])
        self.firstlayer_encode=nn.Sequential(*self.firstlayer_encode)

        #---other layers encode init---#
        self.Encode_layers=[]
        for depth in range(1,self.depth):# 4 layers
            self.onelayer_net=[]
            self.onelayer_net.extend([nn.Conv2d(self.channels_num[depth-1],self.channels_num[depth],kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(self.channels_num[depth]),
                    nn.ReLU()])
            self.onelayer_net.extend([nn.Conv2d(self.channels_num[depth],self.channels_num[depth],kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(self.channels_num[depth]),
                    nn.ReLU()])
            self.onelayer_nets=nn.Sequential(*self.onelayer_net)
            self.Encode_layers.append(self.onelayer_nets)
        # print('Encode_Layers',self.Encode_layers)

        #---other layers decode init---#
        self.Decode_layers=[]
        for depth in range(self.depth-2,-1,-1):# 5 layers
            # print('depth_decode:',depth)
            self.onelayer_net=[]
            self.onelayer_net.extend([nn.Conv2d(self.channels_num[depth+1],self.channels_num[depth],kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm2d(self.channels_num[depth]),
                            nn.ReLU()])
            self.onelayer_net.extend([nn.Conv2d(self.channels_num[depth],self.channels_num[depth],kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(self.channels_num[depth]),
                    nn.ReLU()])
            self.onelayer_nets=nn.Sequential(*self.onelayer_net)
            self.Decode_layers.append(self.onelayer_nets)
        # print('Decode_Layers',self.Decode_layers)
        
        self.Encode_layers=self.__prepare(self.Encode_layers)
        self.Decode_layers=self.__prepare(self.Decode_layers)
        self.Upconv_Layers=self.__prepare(self.Upconv_Layers)
        
    def __prepare(self, net_layers):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if len(net_layers) >= 1:
            return [layer.to(device) for layer in net_layers]
        
    def forward(self,in_img):
        saved_features=[]
        output_layers_img=self.firstlayer_encode(in_img)
        saved_features.append(output_layers_img)
        input_layers_img=self.MaxPooling_Layer(output_layers_img)
        #---encode---#
        for depth in range(self.depth-1):#0, 1, 2, 3
            output_layers_img=self.Encode_layers[depth](input_layers_img)
            saved_features.append(output_layers_img)
            input_layers_img=self.MaxPooling_Layer(output_layers_img)

        
        saved_features.pop()
        #---decode---#
        for depth in range(self.depth-1):#0,1,2,3
            input_layers_img=self.Upconv_Layers[depth](output_layers_img)
            # print('saved_features[len(saved_features)-depth-1]:',saved_features[len(saved_features)-depth-1].size())
            # print('input_layers_img:',input_layers_img.size())
            input_layers_img=torch.cat((saved_features[len(saved_features)-depth-1],input_layers_img),1)#TODO:注意这里的第二个参数，可能会出错
            output_layers_img=self.Decode_layers[depth](input_layers_img)
        
        res=self.finallayer_decode(output_layers_img)
        res+=in_img
        
        return res


class SRCNN(nn.Module):
    def __init__(self,args):
        super(SRCNN, self).__init__()
        #---config init---#
        num_hidden_layers=args.num_hidden_layers
        num_hidden_layers_channel=args.num_hidden_layers_channel
        #---model init---#
        self.net=[]
        self.net.append(nn.Conv2d(3,num_hidden_layers_channel,kernel_size=3,stride=1,padding=1))
        self.net.append(nn.ReLU())
        for i in range(num_hidden_layers):
            self.net.append(nn.Conv2d(num_hidden_layers_channel,num_hidden_layers_channel,kernel_size=3,stride=1,padding=1))
            self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(num_hidden_layers_channel,3,kernel_size=3,stride=1,padding=1))
        self.net.append(nn.ReLU())
        self.net=nn.Sequential(*self.net)

    def forward(self, x):
        res = self.net(x)
        return (res+x)

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats,kernel_size=3,stride=1,padding=1))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

if __name__ == "__main__":
    print(Unet(args))

    # a=torch.ones(2,3,10,10)#2-->batchsize
    # b=torch.ones(2,3,10,10)
    # c=torch.cat((a,b),1)
    # print(c.size())#[2,6,10,10]
