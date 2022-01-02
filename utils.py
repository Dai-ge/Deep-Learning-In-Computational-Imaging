import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io
class SRUtils():
    def __init__(self,output,target,dim=3):

        self.size=output.shape
        self.output=output
        self.target=target
        self.mu_output=output.mean()
        self.mu_target=target.mean()
        self.dev_output=output.std()**2
        self.dev_target=target.std()**2
        self.cov=(np.add(self.output,self.target).std()**2-(self.dev_output+self.dev_target))/2
        
        
        #self.cov_coef=torch.corrcoef(torch.stack((self.output.view(-1),self.target.view(-1)),dim=0))[0,1]
        #self.cov=torch.cov(torch.stack((self.output.view(-1),self.target.view(-1)),dim=0))[0,1]*self.cov_coef
        
        self.k1=0.01
        self.k2=0.03
        self.L=self.output.max()-self.output.min()
        
        self.c1=(self.k1*self.L)**2
        self.c2=(self.k2*self.L)**2

    def RSNR(self):# reconstructed signal-to-noise ratio
        return 20*np.log10(np.linalg.norm(self.target,2,-1).sum()/np.linalg.norm((self.output-self.target),2,-1).sum())
    
    def SSIM(self):
        # return (2*self.mu_output*self.mu_target+self.c1)*(2*self.cov+self.c2)/((self.mu_output**2+self.mu_target**2+self.c1)*(self.dev_output+self.dev_target+self.c2))
        # #TODO: In Astro Porject, change the dimension here, it should be a multichanel image with spatial size 512*512
        # out=self.output.view(self.size).detach().cpu().numpy()
        # target=self.target.view(self.size).detach().cpu().numpy()

        return ssim(self.output,self.target)



if __name__ == "__main__":
    
    out=io.imread("E:\\VScodelib\\Astro\\playground\\BP\\backprojection_ex_multi.png")
    o=SRUtils(out,out)
    print(o.RSNR())

    
