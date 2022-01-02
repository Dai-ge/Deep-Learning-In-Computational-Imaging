import torch.utils.data as data
import os
import torch
##----Not necessary lib----##
import numpy as np
import matplotlib.pyplot as plt
from option import args
from astropy.io import fits

##------Basic Setting------##
TRAIN_HR_DATASET_ROOT_PATH='.\\standalone_project\\datasets\\dataset_single\\new_dataset\\trainingset_fullnumpy'
TEST_HR_DATASET_ROOT_PATH='.\\standalone_project\\datasets\\dataset_single\\new_dataset\\validationset_fullnumpy'
TRAIN_LR_DATASET_ROOT_PATH='.\\standalone_project\\datasets\\dataset_single\\new_dataset\\trainingset_lrnumpy'
TEST_LR_DATASET_ROOT_PATH='.\\standalone_project\\datasets\\dataset_single\\new_dataset\\validationset_lrnumpy'

DEBUG_HR_DATASET_ROOT_PATH='.\\standalone_project\\datasets\\dataset_single\\new_dataset\\debugtest_fullnumpy'
DEBUG_LR_DATASET_ROOT_PATH='.\\standalone_project\\datasets\\dataset_single\\new_dataset\\debugtest_lrnumpy'
##-------------------------##

class AstroData(data.Dataset):
    def __init__(self,args):
        # init configure
        mode=args.mode
        # init data path
        if mode=='train':
            self.HRData_root_path=TRAIN_HR_DATASET_ROOT_PATH
            self.LRData_root_path=TRAIN_LR_DATASET_ROOT_PATH
        elif mode=='test':
            self.HRData_root_path=TEST_HR_DATASET_ROOT_PATH
            self.LRData_root_path=TEST_LR_DATASET_ROOT_PATH
        elif mode=='debug':
            self.HRData_root_path=DEBUG_HR_DATASET_ROOT_PATH
            self.LRData_root_path=DEBUG_LR_DATASET_ROOT_PATH
        else:
            print("The Dataset is not init successfully,maybe the mode is set wrongly...\nThe program is ending here")
            exit()
        
        self.HR_ImgNames=os.listdir(self.HRData_root_path)
        self.LR_ImgNames=os.listdir(self.LRData_root_path)

    def __getitem__(self,idx):
        hr_img_path=os.path.join(self.HRData_root_path,self.HR_ImgNames[idx])
        lr_img_path=os.path.join(self.LRData_root_path,self.LR_ImgNames[idx])
        
        hr_img=fits.getdata(hr_img_path)
        lr_img=fits.getdata(lr_img_path)

        hr_img=np.log10(1000*hr_img+1)/3
        lr_img=lr_img-lr_img.min()
        lr_img=lr_img/lr_img.max()
        
        hr_img_log=torch.from_numpy(hr_img).view(3,512,512)
        lr_img_log=torch.from_numpy(lr_img).view(3,512,512)
        
        # hr_img = torch.unsqueeze(hr_img, dim=0).float()
        # lr_img = torch.unsqueeze(lr_img, dim=0).float()
        
        return hr_img_log,lr_img_log
        

    def __len__(self):
        return len(self.HR_ImgNames)


    def __visualize_fits_data(self):
        ROOT_PATH='D:\\Vscode_proj\\standalone_project\\matlab_files\\examples\\outputs'
        lr_path=os.path.join(ROOT_PATH,'backprojection_ex_superres.fits')
        hr_path=os.path.join('D:\\Vscode_proj\\standalone_project\\matlab_files\\examples\\samples\\single.fits')

        lr_img = fits.getdata(lr_path)
        hr_img = fits.getdata(hr_path)
        print(lr_img)
        print(lr_img.max())
        print(lr_img.min())
        lr_img_log = np.log10(1000*lr_img+1)/3
        hr_img_log = np.log10(1000*hr_img+1)/3
        plt.subplot(1,2,1)
        plt.title('LR_img')
        plt.imshow(lr_img_log)
        plt.subplot(1,2,2)
        plt.title('HR_img')
        plt.imshow(hr_img_log)
        
        plt.show()

if __name__ == "__main__":
    data=AstroData(args)
    for d in data:
        hr,lr=d

        print('lr:',lr.max(),lr.min())
        lr = lr-lr.min()
        lr = lr/lr.max()

        print('lr:',lr)
        print('lr_re:',lr.max(),lr.min())
        
        print('hr:',hr.max(),hr.min())
        hr=np.log10(1000*hr[0].numpy()+1)/3

        # print('lr:',lr)
        print('hr_re:',hr.max(),hr.min())
        break