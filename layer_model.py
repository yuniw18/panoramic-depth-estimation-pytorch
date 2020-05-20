import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
import random
import numpy as np
import os




def padding(x, p, ring_pad = True):
    ############# Tensorflow function ##################

    p = p
    if ring_pad:
        p_x = F.pad(x,(0,0,p,p ))
        left = p_x[:,:,:,:p]
        right = p_x[:,:,:,-p:]
 
        p_x = torch.cat((right,p_x,left),3)

    else:
        p_x = F.pad(x,(p,p,p,p))
    return p_x


def conv(in_channel, out_channel, kernel_size, stride, ring_pad = True):
######################### Tensorflow function #############################
    
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)  ## OK


    if ring_pad:

         return nn.Conv2d(in_channel, out_channel,kernel_size,stride)

    else:
    
        return nn.Conv2d(in_channel, out_channel,kernel_size,stride,padding=p)

def upconv(in_channel, out_channel, kernel_size, scale,ring_pad = True):
###################### Tensorflow function ################################
#    upsample = nn.Upsample(scale_factor=scale)

    p = np.floor((kernel_size - 1) / 2).astype(np.int32)  ## OK

    if ring_pad:
        conv = nn.Conv2d(in_channel, out_channel, kernel_size, 1)
        
    else:
        conv = nn.Conv2d(in_channel, out_channel, kernel_size, 1,padding=p)
    return conv

def get_disp(in_channel,ring_pad=True):
    kernel_size=3
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)  ## OK
    if ring_pad:
        disp = nn.Conv2d(in_channel, 2, kernel_size=3, stride=1)
    else:
        disp = nn.Conv2d(in_channel, 2, kernel_size=3, stride=1,padding=p)
    return disp


class VGG_128(nn.Module):
    def __init__(self):
        super(VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv1 = conv(3, 32, kernel_size=7, stride=1)

        self.conv2 = conv(32, 32, kernel_size=7, stride=2)
        # [64, 64, 64]
        self.conv3 = conv(32, 64, kernel_size=5, stride=1)
        self.conv4 = conv(64, 64, kernel_size=5, stride=2)
        # [128, 32, 32]
        self.conv5 = conv(64, 128, kernel_size=3, stride=1)
        self.conv6 = conv(128, 128, kernel_size=3, stride=2)
        
        self.conv7 = conv(128, 256, kernel_size=3, stride=1)
        self.conv8 = conv(256, 256, kernel_size=3, stride=2)
        
        self.conv9 = conv(256, 512, kernel_size=3, stride=1)
        self.conv10 = conv(512, 512, kernel_size=3, stride=2)
        
        self.conv11 = conv(512, 512, kernel_size=3, stride=1)
        self.conv12 = conv(512, 512, kernel_size=3, stride=2)
        
        self.conv13 = conv(512, 512, kernel_size=3, stride=1)
        self.conv14 = conv(512, 512, kernel_size=3, stride=2)

##########################################################################################################

        self.upconv1 = upconv(512,512,kernel_size=3,scale=2) #upconv7
        self.upconv2 = conv(1024,512,kernel_size=3,stride=1)

        self.upconv3 = upconv(512,512,kernel_size=3,scale=2) #upconv6
        self.upconv4 = conv(1024,512,kernel_size=3,stride=1)

        self.upconv5 = upconv(512,256,kernel_size=3,scale=2) #upconv5
        self.upconv6 = conv(512,256,kernel_size=3,stride=1)

        self.upconv7 = upconv(256,128,kernel_size=3,scale=2) #upconv4
        self.upconv8 = conv(256,128,kernel_size=3,stride=1)

        self.upconv9 = upconv(128,64,kernel_size=3,scale=2) #upconv3
        self.upconv10 = conv(130,64,kernel_size=3,stride=1)

        self.upconv11 = upconv(64,32,kernel_size=3,scale=2) #upconv2
        self.upconv12 = conv(66,32,kernel_size=3,stride=1)

        self.upconv13 = upconv(32,16,kernel_size=3,scale=2) #upconv1
        self.upconv14 = conv(18,16,kernel_size=3,stride=1)

        self.upsample = nn.Upsample(scale_factor=2)
 
        self.get_disp1 = get_disp(128)
        self.get_disp2 = get_disp(64)
        self.get_disp3 = get_disp(32)
        self.get_disp4 = get_disp(16)


        self.elu = nn.ELU()
        self.sig = nn.Sigmoid()

    def load_weights(self,weight_path):

        self.conv1.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv.weights:0.npy")).reshape(7,7,3,32)).float().permute(3,2,0,1)
        self.conv1.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv.biases:0.npy"))).float().squeeze()

        self.conv2.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_1.weights:0.npy")).reshape(7,7,32,32)).float().permute(3,2,0,1)
        self.conv2.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_1.biases:0.npy"))).float().squeeze()

        self.conv3.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_2.weights:0.npy")).reshape(5,5,32,64)).float().permute(3,2,0,1)
        self.conv3.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_2.biases:0.npy"))).float().squeeze()

        self.conv4.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_3.weights:0.npy")).reshape(5,5,64,64)).float().permute(3,2,0,1)
        self.conv4.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_3.biases:0.npy"))).float().squeeze()

        self.conv5.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_4.weights:0.npy")).reshape(3,3,64,128)).float().permute(3,2,0,1)
        self.conv5.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_4.biases:0.npy"))).float().squeeze()

        self.conv6.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_5.weights:0.npy")).reshape(3,3,128,128)).float().permute(3,2,0,1)
        self.conv6.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_5.biases:0.npy"))).float().squeeze()

        self.conv7.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_6.weights:0.npy")).reshape(3,3,128,256)).float().permute(3,2,0,1)
        self.conv7.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_6.biases:0.npy"))).float().squeeze()

        self.conv8.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_7.weights:0.npy")).reshape(3,3,256,256)).float().permute(3,2,0,1)
        self.conv8.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_7.biases:0.npy"))).float().squeeze()

        self.conv9.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_8.weights:0.npy")).reshape(3,3,256,512)).float().permute(3,2,0,1)
        self.conv9.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_8.biases:0.npy"))).float().squeeze()

        self.conv10.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_9.weights:0.npy")).reshape(3,3,512,512)).float().permute(3,2,0,1)
        self.conv10.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_9.biases:0.npy"))).float().squeeze()

        self.conv11.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_10.weights:0.npy")).reshape(3,3,512,512)).float().permute(3,2,0,1)
        self.conv11.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_10.biases:0.npy"))).float().squeeze()

        self.conv12.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_11.weights:0.npy")).reshape(3,3,512,512)).float().permute(3,2,0,1)
        self.conv12.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_11.biases:0.npy"))).float().squeeze()

        self.conv13.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_12.weights:0.npy")).reshape(3,3,512,512)).float().permute(3,2,0,1)
        self.conv13.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_12.biases:0.npy"))).float().squeeze()

        self.conv14.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_13.weights:0.npy")).reshape(3,3,512,512)).float().permute(3,2,0,1)
        self.conv14.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.encoder.Conv_13.biases:0.npy"))).float().squeeze()
#############################################################################################################################################################################################
        self.upconv1.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv.weights:0.npy")).reshape(3,3,512,512)).float().permute(3,2,0,1)
        self.upconv1.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv.biases:0.npy"))).float().squeeze()

        self.upconv2.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_1.weights:0.npy")).reshape(3,3,1024,512)).float().permute(3,2,0,1)
        self.upconv2.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_1.biases:0.npy"))).float().squeeze()

        self.upconv3.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_2.weights:0.npy")).reshape(3,3,512,512)).float().permute(3,2,0,1)
        self.upconv3.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_2.biases:0.npy"))).float().squeeze()

        self.upconv4.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_3.weights:0.npy")).reshape(3,3,1024,512)).float().permute(3,2,0,1)
        self.upconv4.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_3.biases:0.npy"))).float().squeeze()

        self.upconv5.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_4.weights:0.npy")).reshape(3,3,512,256)).float().permute(3,2,0,1)
        self.upconv5.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_4.biases:0.npy"))).float().squeeze()

        self.upconv6.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_5.weights:0.npy")).reshape(3,3,512,256)).float().permute(3,2,0,1)
        self.upconv6.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_5.biases:0.npy"))).float().squeeze()

        self.upconv7.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_6.weights:0.npy")).reshape(3,3,256,128)).float().permute(3,2,0,1)
        self.upconv7.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_6.biases:0.npy"))).float().squeeze()

        self.upconv8.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_7.weights:0.npy")).reshape(3,3,256,128)).float().permute(3,2,0,1)
        self.upconv8.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_7.biases:0.npy"))).float().squeeze()
    
        self.get_disp1.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_8.weights:0.npy")).reshape(3,3,128,2)).float().permute(3,2,0,1)
        self.get_disp1.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_8.biases:0.npy"))).float().squeeze()
 
        self.upconv9.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_9.weights:0.npy")).reshape(3,3,128,64)).float().permute(3,2,0,1)
        self.upconv9.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_9.biases:0.npy"))).float().squeeze()

        self.upconv10.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_10.weights:0.npy")).reshape(3,3,130,64)).float().permute(3,2,0,1)
        self.upconv10.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_10.biases:0.npy"))).float().squeeze()

        self.get_disp2.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_11.weights:0.npy")).reshape(3,3,64,2)).float().permute(3,2,0,1)
        self.get_disp2.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_11.biases:0.npy"))).float().squeeze()
 
        self.upconv11.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_12.weights:0.npy")).reshape(3,3,64,32)).float().permute(3,2,0,1)
        self.upconv11.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_12.biases:0.npy"))).float().squeeze()

        self.upconv12.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_13.weights:0.npy")).reshape(3,3,66,32)).float().permute(3,2,0,1)
        self.upconv12.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_13.biases:0.npy"))).float().squeeze()

        self.get_disp3.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_14.weights:0.npy")).reshape(3,3,32,2)).float().permute(3,2,0,1)
        self.get_disp3.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_14.biases:0.npy"))).float().squeeze()

        self.upconv13.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_15.weights:0.npy")).reshape(3,3,32,16)).float().permute(3,2,0,1)
        self.upconv13.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_15.biases:0.npy"))).float().squeeze()

        self.upconv14.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_16.weights:0.npy")).reshape(3,3,18,16)).float().permute(3,2,0,1)
        self.upconv14.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_16.biases:0.npy"))).float().squeeze()

        self.get_disp4.weight.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_17.weights:0.npy")).reshape(3,3,16,2)).float().permute(3,2,0,1)
        self.get_disp4.bias.data = torch.from_numpy(np.load(os.path.join(weight_path,"model.decoder.Conv_17.biases:0.npy"))).float().squeeze()

    def forward(self, x,ring_pad,kitti):
        
        out = padding(x,np.floor((7 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv1(out)
        out = self.elu(out)
        out = padding(out,np.floor((7 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv2(out)
        out = self.elu(out)
        skip1 = out
         
        out = padding(out,np.floor((5 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv3(out)
        out = self.elu(out)
        out = padding(out,np.floor((5 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv4(out)
        out = self.elu(out)
        skip2 = out

        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv5(out)
        out = self.elu(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv6(out)
        out = self.elu(out)
        skip3 = out

 
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv7(out)
        out = self.elu(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv8(out)
        out = self.elu(out)
        skip4 = out

 
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv9(out)
        out = self.elu(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv10(out)
        out = self.elu(out)
        skip5 = out


 
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv11(out)
        out = self.elu(out)
        
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv12(out)
        out = self.elu(out)
        skip6 = out

        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv13(out)
        out = self.elu(out)

        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.conv14(out)
        out = self.elu(out)
        debug = out
##############################################################################
        
        out = self.upsample(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv1(out)
        out = self.elu(out)
        out = torch.cat((out,skip6),1)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        
        out = self.upconv2(out)
        out = self.elu(out)
        
        out = self.upsample(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv3(out)
        out = self.elu(out)
        out = torch.cat((out,skip5),1)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv4(out)
        out = self.elu(out)


        out = self.upsample(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv5(out)
        out = self.elu(out)
        out = torch.cat((out,skip4),1)
        
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv6(out)
        out = self.elu(out)

        out = self.upsample(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv7(out)
        out = self.elu(out)
        out = torch.cat((out,skip3),1)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv8(out)
        out = self.elu(out)
        disp1 = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        disp1 = self.get_disp1(disp1)
        disp1 = 0.3 * self.sig(disp1)
        udisp1 = self.upsample(disp1)


        out = self.upsample(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv9(out)
        out = self.elu(out)
        out = torch.cat((out,skip2,udisp1),1)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv10(out)
        out = self.elu(out)
        disp2 = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        disp2 = self.get_disp2(disp2)
        disp2 = 0.3 * self.sig(disp2)
        udisp2 = self.upsample(disp2)


        out = self.upsample(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv11(out)
        out = self.elu(out)
        out = torch.cat((out,skip1,udisp2),1)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv12(out)
        out = self.elu(out)
        disp3 = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        disp3 = self.get_disp3(disp3)
        disp3 = 0.3 * self.sig(disp3)
        udisp3 = self.upsample(disp3)


        out = self.upsample(out)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv13(out)
        out = self.elu(out)
        out = torch.cat((out,udisp3),1)
        out = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        out = self.upconv14(out)
        out = self.elu(out)
        disp4 = padding(out,np.floor((3 - 1) / 2).astype(np.int32),ring_pad)
        disp4 = self.get_disp4(disp4)

 
        if kitti:
            disp4 = 0.3 * self.sig(disp4)
            return disp4,disp3,disp2,disp1
        else:
            return disp4,disp3,disp2,disp1
