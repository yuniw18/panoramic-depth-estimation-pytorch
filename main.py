import argparse
import os
from trainer import Train
from torch.backends import cudnn
from torch.utils.data import DataLoader
from data_load import KITTI_loader
from torchvision import transforms
import torch
import ThreeD60
def main(config):
    cudnn.benchmark = True
    torch.manual_seed(1593665876)
    torch.cuda.manual_seed_all(4099049913103886)    
     
    transform = transforms.Compose([
                    transforms.Resize((config.input_height,config.input_width)),
                    transforms.ToTensor(),
                    ])

    ThreeD_dataloader = None
    kitti_dataloader = None

    if config.ThreeD == True:
        ThreeD_loader = ThreeD60.get_datasets(config.train_path, \
                datasets=["suncg","m3d", "s2d3d"],
                placements=[ThreeD60.Placements.CENTER,ThreeD60.Placements.RIGHT,ThreeD60.Placements.UP],
                image_types=[ThreeD60.ImageTypes.COLOR, ThreeD60.ImageTypes.DEPTH, ThreeD60.ImageTypes.NORMAL], longitudinal_rotation=True)
        ThreeD_dataloader = DataLoader(ThreeD_loader,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers)
    if config.KITTI == True:
        kitti_loader = KITTI_loader(config.kitti_train_path,transform)
        kitti_dataloader = DataLoader(kitti_loader,batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers)
        


 
    if config.mode == 'train':
        train = Train(config,ThreeD_dataloader,kitti_dataloader)
        train.train()

    elif config.mode == 'sample':
        train = Train(config,ThreeD_dataloader,kitti_dataloader)
        eval_name = 'evaluation'
        train.evaluate(config.val_path,config.checkpoint_path,eval_name)
    
    elif config.mode == 'make':
        train = Train(config,ThreeD_dataloader,kitti_dataloader)
        train.make_checkpoints() 
 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--network_type',type=str,default='RectNet') # RectNet, UResNet
    parser.add_argument('--experiment_name',type=str,default='omnidepth') # path where checkpoint will be saved in 'experiments'

    parser.add_argument('--train_path',type=str,default='../3D60/splits/filtered/final_train2.txt') # text which contains train image list
    parser.add_argument('--kitti_train_path',type=str,default='../3D60/splits/filtered/final_train2.txt') # text which contains train image list
 
    parser.add_argument('--val_path',type=str,default='./SAMPLE') # text which contains valid image list
    parser.add_argument('--test_path',type=str,default='../3D60/splits/filtered/final_test.txt') # text which contains test image list

    parser.add_argument('--test_folder_path',type=str,default='./demo_test/') # path where test image exist, used in mode:2


    parser.add_argument('--output_path',type=str,default='./demo_results') # path where inferenced image will be saved, used in mode:2
    
    parser.add_argument('--checkpoint_path',type=str,default='./pre_trained/generator.pkl') # path where checkpoint to load exists
    parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
    parser.add_argument('--rectilinear_mode', help='rectilinear mode', action='store_true')
    parser.add_argument('--KITTI', help='KITTI', action='store_true')
    parser.add_argument('--ThreeD', help='ThreeD', action='store_true')
    parser.add_argument('--AAGAN', help='AAGAN', action='store_true')

    parser.add_argument('--fov', type=float, help='Horizontal field of view (in degrees)', default=82.5)
    parser.add_argument('--input_height',              type=int,   help='input height', default=256)
    parser.add_argument('--input_width',               type=int,   help='input width', default=512)

    parser.add_argument('--kitti_width', type=int, default=512)
    parser.add_argument('--kitti_height', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='./checkpoints/default')
     
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--eval_path', type=str, default='evaluate')

    parser.add_argument('--image_path', type=str, default='./input/')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)
    parser.add_argument('--checkpoint_step', type=int , default=500)


#    parser.add_argument('--test_path', type=str, default='./testset/')
    
    config = parser.parse_args()
    
    print(config)
    main(config)
