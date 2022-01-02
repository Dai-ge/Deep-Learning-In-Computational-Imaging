import torch
import torch.nn as nn


class SRUtils():
    def __init__(self):
        pass
    
    def RSNR(self,output,target):# reconstructed signal-to-noise ratio
        return 20*torch.log10(torch.norm(target,2)/torch.norm(torch.sub(output,target),2))

    def ISNR(self,target,sigma_Noise):#input signal-to-noise ratio
        return 20*torch.log10(torch.norm(target,2)/(torch.sqrt(2)*sigma_Noise))


if __name__ == "__main__":
    out=torch.Tensor([[1,2,3],[4,5,6]])
    target=torch.Tensor([[7,8,9],[11,2,3]])
    o=SRUtils(out,target)
    test=torch.ones(3)
    test2=torch.ones(3)
    print(o.RSNR().item())
    
