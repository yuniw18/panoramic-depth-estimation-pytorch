import math
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import os
from layer_model import VGG_128 
from PIL import Image
import numpy as np
import torch.nn as nn
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from bilinear import *
from criteria import L2Loss
from torch import optim
import ThreeD60
from torch.autograd import Variable
import OpenEXR
import Imath
import array
class Train(object):
    def __init__(self,config,data_loader,kitti_loader):
        self.vgg = None
        self.vgg_eval = None
        self.checkpoint_path = config.checkpoint_path
        self.eval_data_path = config.val_path
        self.l2_loss = L2Loss()
        self.model_name = config.model_name
        self.model_path = os.path.join(config.model_name,config.model_path)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lr = config.lr 
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.KITTI = config.KITTI
        self.ThreeD = config.ThreeD
        self.AAGAN = config.AAGAN
        self.sample_path = os.path.join(self.model_name,config.sample_path)
        self.log_path = os.path.join(self.model_name,'log.txt')
        self.eval_path = os.path.join(self.model_name,config.eval_path)
        self.data_loader = data_loader
        self.kitti_loader = kitti_loader
        self.num_epochs = config.num_epochs
        self.max_depth = 255.0
        self.batch_size = config.batch_size
        self.config = config
        
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.sample_path):
            os.mkdir(self.sample_path)
        if not os.path.exists(self.eval_path):
            os.mkdir(self.eval_path)


        if config.ThreeD == True:
            with open(config.train_path,'r') as f:
                num_samples = len(f.readlines())
        elif config.KITTI == True:
            num_samples = len(self.kitti_loader)
            print(num_samples)
        self.num_samples = num_samples

        self.build_model()
    def build_model(self):
        self.vgg = VGG_128()
        self.vgg_eval = VGG_128()
        self.g_optimizer = optim.Adam(self.vgg.parameters(),
                                        self.lr,[self.beta1,self.beta2])

        if torch.cuda.is_available():
            self.vgg.cuda()
            self.vgg_eval.cuda()
    def to_variable(self,x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    def transform(self,input):
        transform = transforms.Compose([
                    transforms.ToTensor()])
        return transform(input)
    def gray_transform(self,input):
        transform = transforms.Compose([transforms.ToPILImage(),
                    transforms.Grayscale(),
                    transforms.ToTensor()])
        return transform(input)
 
    def read_exr(self, image_fpath):
        f = OpenEXR.InputFile( image_fpath )
        dw = f.header()['dataWindow']
        w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
        im = np.empty( (h, w, 3) )

        # Read in the EXR
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = f.channels( ["R", "G", "B"], FLOAT )
        for i, channel in enumerate( channels ):
            im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
        return im


    def parse_data(self, data):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output, and a list of the remaining info as a third output. Must be implemented.
        '''
        rgb = data[0].to(self.device)
        gt_depth_1x = data[1].to(self.device)
        gt_depth_2x = F.interpolate(gt_depth_1x, scale_factor=0.5)
        gt_depth_4x = F.interpolate(gt_depth_1x, scale_factor=0.25)
        mask_1x = data[2].to(self.device)
        mask_2x = F.interpolate(mask_1x, scale_factor=0.5)
        mask_4x = F.interpolate(mask_1x, scale_factor=0.25)

        inputs = [rgb]
        gt = [gt_depth_1x, mask_1x, gt_depth_2x, mask_2x, gt_depth_4x, mask_4x]

        return inputs, gt


    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x  = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y , 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def reset_grad(self):
        self.vgg.zero_grad()
    def post_process_disparity(self,disp):
        disp = disp.cpu().detach().numpy() 
        _, h, w = disp.shape
        l_disp = disp[0,:,:]
        r_disp = np.fliplr(disp[1,:,:])
        m_disp = 0.5 * (l_disp + r_disp)
        return m_disp
    def gradient_x(self,img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:,:,:,:-1] - img[:,:,:,1:]
        return gx

    def gradient_y(self,img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:,:,:-1,:] - img[:,:,1:,:]
        return gy

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(4)]

    def normalize(self,input):
        transform_t = transforms.Compose([
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        return transform_t(input)
    def print_progress(self, batch_num, max_batch_num, batch_size, kitti_loss,ThreeD_loss, log_path,epoch,max_arrow=50):
        self.epoch = epoch
        self.i = (batch_num + 1) * batch_size if batch_num < max_batch_num else self.num_samples
        num_arrow = int(self.i * max_arrow / self.num_samples)
        num_line = max_arrow - num_arrow  
        percent = self.i * 100.0 / self.num_samples 
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%' + '\r'  
        if True:
            print('\r',
              'Epoch: [{0}][{1}/{2}]'.format(self.epoch + 1, batch_num + 1,
                                             len(self.kitti_loader)) + ' - KITTI_Loss: %.5f ' % kitti_loss.item() +' - ThreeD_Loss: %.3f ' % ThreeD_loss.item() ,
              end='') 
        f = open(log_path,'a')
        f.write('Epoch: [{0}][{1}/{2}]'.format(self.epoch + 1, batch_num + 1,
                                             len(self.kitti_loader)) + ' - KITTI_Loss: %.5f ' % kitti_loss.item() +' - ThreeD_Loss: %.3f ' % ThreeD_loss.item() + '\n' ) 
        f.close ()
        if self.i >= self.num_samples:
            print('')

    def generate_image_left(self, img, disp,config):
        if not config.rectilinear_mode:
            return bilinear_sampler_equirectangular(img.permute(0,2,3,1), -disp.permute(0,2,3,1), config.fov)
        else:
            return bilinear_sampler_1d_h(img.permute(0,2,3,1), -disp.permute(0,2,3,1)).permute(0,3,1,2)

    def generate_image_right(self, img, disp,config):
        if not config.rectilinear_mode:
            return bilinear_sampler_equirectangular(img, disp, config.fov)
        else:
            return bilinear_sampler_1d_h(img.permute(0,2,3,1), disp.permute(0,2,3,1)).permute(0,3,1,2)
    def save_samples(self, inputs, gt, mask, outputs,sample_path,epoch,batch_num):
        torchvision.utils.save_image(inputs.data,os.path.join(sample_path,'input_samples-%d-%d.png' %(epoch, batch_num)))
        torchvision.utils.save_image((gt.data),os.path.join(sample_path,'GT_samples-%d-%d.png' %(epoch, batch_num)))
        torchvision.utils.save_image((outputs.data),os.path.join(sample_path,'output_samples-%d-%d.png' %(epoch, batch_num)))
        torchvision.utils.save_image((mask.data),os.path.join(sample_path,'mask_samples-%d-%d.png' %(epoch, batch_num)))

    def scale_pyramid(self,img,num_scales):
        scaled_imgs = [img]
        height = img.size(2)
        width = img.size(3) 
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            scaled_imgs.append(F.interpolate(img,size=(height//ratio,width//ratio)))
        return scaled_imgs
    def make_checkpoints(self):
        self.vgg.load_weights('./numpy_weight')
        self.vgg.cuda()
        g_path = os.path.join(self.model_path,'generator.pkl')
        torch.save(self.vgg.state_dict(),g_path)
 
    def train(self):
        if os.path.isfile(self.log_path):
            os.remove(self.log_path)  

        self.vgg.load_state_dict(torch.load(self.checkpoint_path))
        self.max_depth = 255.0
        
        max_batch_num = len(self.kitti_loader) - 1
        for epoch in range(self.num_epochs):
            for batch_num, data in enumerate(self.kitti_loader):
        
########################################### ThreeD60 dataset training ############################################        
                if (self.KITTI == False and self.ThreeD == False):
                    print("Plz check dataset for training")
                    break
                if self.ThreeD:
                    data = next(iter(self.data_loader))
                    upsample = nn.Upsample((256,512))
               
                    inputs = self.to_variable(ThreeD60.extract_image(data,ThreeD60.Placements.CENTER,ThreeD60.ImageTypes.COLOR))

                    gt = self.to_variable(ThreeD60.extract_image(data,ThreeD60.Placements.CENTER,ThreeD60.ImageTypes.DEPTH))

                    mask = self.to_variable(((gt <= self.max_depth) & (gt > 0.)).to(torch.float32))
                
                    gt = gt*mask

                    output = self.vgg(inputs,ring_pad=True,kitti=False)
                    
                    output = output[:,0,:,:].unsqueeze(1)
        
                    ThreeD_loss = self.l2_loss(output,gt,mask) 
                    self.reset_grad()
                    ThreeD_loss.backward()
                    self.g_optimizer.step()
 
##################################### KITTI dataset training #############################################
                if self.KITTI:
                    data_kitti = next(iter(self.kitti_loader))
                    left = self.to_variable(data_kitti[0])
                    right = self.to_variable(data_kitti[1])

              
                    left_pyramid = self.scale_pyramid(left,4)
                    right_pyramid = self.scale_pyramid(right,4)
                 
                    kitti_output = self.vgg(left,ring_pad=False,kitti=True) 
                    disp_left_est = [d[:,0,:,:].unsqueeze(1) for d in kitti_output]
                    disp_right_est = [d[:,1,:,:].unsqueeze(1) for d in kitti_output]
        
                #generate images
                
                    left_est = [self.generate_image_left(right_pyramid[i],disp_left_est[i],self.config) for i in range(4)] 
                    right_est = [self.generate_image_right(left_pyramid[i],disp_right_est[i],self.config) for i in range(4) ]

                #LR consistenry
                    right_to_left_disp = [self.generate_image_left(disp_right_est[i],disp_left_est[i],self.config) for i in range(4)] 
                    left_to_right_disp = [self.generate_image_right(disp_left_est[i],disp_right_est[i],self.config) for i in range(4)] 
                # disparity smoothness
                    disp_left_smoothness = self.get_disparity_smoothness(disp_left_est,left_pyramid)
                    disp_right_smoothness = self.get_disparity_smoothness(disp_right_est,right_pyramid)
                
                ########## buliding losses #########
                 
                    l1_left = [torch.abs(left_est[i] - left_pyramid[i]) for i in range(4)]
                    l1_reconstruction_loss_left = [torch.mean(l) for l in l1_left]
                    l1_right = [torch.abs(right_est[i] - right_pyramid[i]) for i in range(4)]
                    l1_reconstruction_loss_right = [torch.mean(l) for l in l1_right]
##############################################################################################################
                    ssim_left = [self.SSIM(left_est[i],  left_pyramid[i]) for i in range(4)]
                    ssim_loss_left  = [torch.mean(s) for s in ssim_left]
                    ssim_right = [self.SSIM(right_est[i], right_pyramid[i]) for i in range(4)]
                    ssim_loss_right = [torch.mean(s) for s in ssim_right]

            # WEIGTHED SUM
    
                    image_loss_right = [self.config.alpha_image_loss * ssim_loss_right[i] + (1 - self.config.alpha_image_loss) * l1_reconstruction_loss_right[i] for i in range(4)]
                    image_loss_left  = [self.config.alpha_image_loss * ssim_loss_left[i]  + (1 - self.config.alpha_image_loss) * l1_reconstruction_loss_left[i]  for i in range(4)]

                    image_loss = (image_loss_left + image_loss_right)

            # DISPARITY SMOOTHNESS
                    disp_left_loss  = [torch.mean(torch.abs(disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
                    disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(4)]

                    disp_gradient_loss = disp_left_loss + disp_right_loss

            # LR CONSISTENCY
                    lr_left_loss  = [torch.mean(torch.abs(right_to_left_disp[i] - disp_left_est[i]))  for i in range(4)]
                    lr_right_loss = [torch.mean(torch.abs(left_to_right_disp[i] - disp_right_est[i])) for i in range(4)]
                    lr_loss = lr_left_loss + lr_right_loss
                    kitti_loss = 0
            # TOTAL LOSS
                    
                    for i in range(4):
                        kitti_loss += image_loss[i] + self.config.disp_gradient_loss_weight * disp_gradient_loss[i] +  self.config.lr_loss_weight * lr_loss[i]
                    total_loss = 0 
                if self.KITTI and self.ThreeD:
                    total_loss = ThreeD_loss + kitti_loss
                elif self.KITTI and not self.ThreeD:
                    total_loss =  kitti_loss
                elif self.ThreeD and not self.KITTI: 
                    total_loss = ThreeD_loss
 
                self.reset_grad()
                total_loss.backward()
                self.g_optimizer.step()

                                
            
                if (batch_num) % self.log_step == 0:
                    if self.KITTI == False:
                        kitti_loss = torch.FloatTensor([0])
                    if self.ThreeD == False:
                        ThreeD_loss = torch.FloatTensor([0])
                    self.print_progress(batch_num, max_batch_num, self.batch_size, kitti_loss,ThreeD_loss,self.log_path,epoch)
                if (batch_num) % self.sample_step == 0:
                    if self.ThreeD:
                        self.save_samples(inputs,gt/10,mask,output/10,self.sample_path,self.epoch + 1,batch_num)
                if (batch_num) % self.config.checkpoint_step == 0:
                    g_path = os.path.join(self.model_path,'generator-%d-%d.pkl' % (epoch + 1,batch_num))
                    torch.save(self.vgg.state_dict(),g_path)
                    eval_name = '3d60_%d-%d' %(epoch+1,batch_num)
                    self.evaluate(self.eval_data_path,g_path,eval_name)
            
############# test function #############
    def evaluate(self,root,checkpoint_path,eval_name):
        L1_loss = nn.L1Loss()
        self.vgg_eval.load_state_dict(torch.load(checkpoint_path))
        image_list = os.listdir(root)
        eval_image = []
        for image_name in image_list:
            eval_image.append(os.path.join(root,image_name))


        index = 0  
        for image_path in eval_image:
            stereo_baseline = 0.472
            index = index + 1
 
            input_image = scipy.misc.imread(image_path, mode="RGB")
         
            original_height, original_width, num_channels = input_image.shape
        
            input_height = 256
            input_width = 1024

         
            minor_height = int(input_width/input_image.shape[1] * input_image.shape[0]/0.7191)
            input_image = scipy.misc.imresize(input_image, [minor_height, input_width], interp='lanczos')
        
            input_image = np.pad(input_image, ((input_height-minor_height,0),(0,0),(0,0)), mode='constant')
            input_image = input_image.astype(np.float32) / 255.0
        
            input_image = np.stack((input_image, np.fliplr(input_image)), 0)
            input_image = torch.from_numpy(input_image).float().permute(0,3,1,2).cuda()

            disp = self.vgg_eval(input_image,ring_pad=True,kitti=True)
        
            disp_comp = disp[0]
            disp_comp = disp_comp[:,0,:,:]


            disp_pp = self.post_process_disparity(disp_comp).astype(np.float32)
        
            disp_pp = disp_pp[input_height-minor_height:]

            pred_width = disp_pp.shape[1]
            disp_pp = cv2.resize(disp_pp.squeeze(), (original_width, original_height))

            angular_precision = original_width/math.pi
            disp_pp *= angular_precision/stereo_baseline*original_width/pred_width

            save_name = eval_name + '_'+str(index)+'.png'        
            plt.imsave(os.path.join(self.eval_path,save_name ), disp_pp, cmap='viridis')

       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
               
