import math
import torch
import torch.nn.functional as F


def bilinear_sampler_1d_h(input_images, x_offset,  wrap_mode='border', name='bilinear_sampler'):
    _num_batch    = input_images.size(0)
    _height       = input_images.size(1)
    _width        = input_images.size(2)
    _num_channels = input_images.size(3)

    _height_ft = torch.cuda.FloatTensor([float(_height)])
    _width_ft = torch.cuda.FloatTensor([float(_width)])

    _height_f = float(_height)
    _width_f = float(_width)
    _wrap_mode = wrap_mode

    def _repeat(x, n_repeats):

        rep = x.unsqueeze(1).repeat(1,n_repeats).view(-1) #check unsqueeze x is right
        return rep

    def _interpolate(im,x,y):
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size=1
            im = F.pad(im,(0,0,1,1,1,1,0,0))
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = torch.clamp(x, 0.0, _width_f - 1 + 2 * _edge_size)

        x0_f = x.floor()
        y0_f = y.floor()
        x1_f = x0_f + 1
        

        x0 = x0_f.long()
        y0 = y0_f.long()
        x1 = torch.min(x1_f,torch.cuda.FloatTensor([_width_f -1 + 2 * _edge_size])).long()

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(torch.arange(_num_batch) * dim1, _height * _width).long()
        base = base.cuda()
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = im.view(-1,_num_channels)

        pix_l = torch.index_select(im_flat,0,idx_l)
        pix_r = torch.index_select(im_flat,0,idx_r)


        weight_l = (x1_f - x).unsqueeze(1)
        weight_r = (x - x0_f).unsqueeze(1)

        return pix_l * weight_l + weight_r * pix_r

    def _transform(input_images,x_offset):
   
         x_t, y_t = torch.meshgrid(torch.linspace(0.0,_width_f - 1.0, _width),
                                    torch.linspace(0.0, _height_f - 1.0, _height))
         
         x_t = x_t.permute(1,0).cuda()
         y_t = y_t.permute(1,0).cuda()
         
    
         x_t_flat = x_t.contiguous().view(1,-1).repeat(_num_batch,1)
         y_t_flat = y_t.contiguous().view(1,-1).repeat(_num_batch,1)

         x_t_flat = x_t_flat.view(-1)
         y_t_flat = y_t_flat.view(-1)

         x_t_flat = x_t_flat + x_offset.contiguous().view(-1) * _width_f   

         input_transformed = _interpolate(input_images,x_t_flat,y_t_flat)

         output = input_transformed.view(_num_batch,_height,_width,_num_channels)
         
         return output

    output = _transform(input_images, x_offset)
 
    return output

def bilinear_sampler_equirectangular(input_images, d_offset, fov, wrap_mode='border', name='bilinear_sampler', **kwargs):
    _num_batch    = input_images.size(0)
    _height       = input_images.size(1)
    _width        = input_images.size(2)
    _num_channels = input_images.size(3)

#    _height_ft = torch.cuda.FloatTensor([float(_height)])
#    _width_ft = torch.cuda.FloatTensor([float(_width)])

    _height_f = float(_height)
    _width_f = float(_width)
    _wrap_mode = wrap_mode


    def _repeat(x, n_repeats):

        rep = x.unsqueeze(1).repeat(1,n_repeats).view(-1) 
        return rep

    def _interpolate(im,x,y):
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size=1
            im = F.pad(im,(0,0,1,1,1,1,0,0))
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = torch.clamp(x, 0.0, _width_f - 1 + 2 * _edge_size)

        x0_f = x.floor()
        y0_f = y.floor()
        x1_f = x0_f + 1
        

        x0 = x0_f.long()
        y0 = y0_f.long()
        x1 = torch.min(x1_f,torch.cuda.FloatTensor([_width_f -1 + 2 * _edge_size])).long()

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(torch.arange(_num_batch) * dim1, _height * _width).long()
        base = base.cuda()
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = im.view(-1,_num_channels)

################### check this part  #################################
        idx_l = torch.clamp(idx_l,min =0, max = im_flat.size(0)-1)
        idx_r = torch.clamp(idx_r,min =0, max = im_flat.size(0)-1)
#####################################################################

        pix_l = torch.index_select(im_flat,0,idx_l)
        pix_r = torch.index_select(im_flat,0,idx_r)


        weight_l = (x1_f - x).unsqueeze(1)
        weight_r = (x - x0_f).unsqueeze(1)

        return pix_l * weight_l + weight_r * pix_r

    def _transform(input_images,d_offset):
         angular_precision = _width_f/(fov*math.pi/180)
         P = torch.cuda.FloatTensor([angular_precision, 0., _width_f/2.,
                     0., angular_precision, _height_f/2.,
                     0.,                0., 1.])



         P = P.view(3,3)
         Pinv = torch.inverse(P)

         x_t, y_t = torch.meshgrid(torch.linspace(0.0,_width_f - 1.0, _width),
                                    torch.linspace(0.0, _height_f - 1.0, _height))
         
         x_t = x_t.permute(1,0).cuda()
         y_t = y_t.permute(1,0).cuda()
         
    
         x_t_flat = x_t.contiguous().view(1,-1).repeat(_num_batch,1)
         y_t_flat = y_t.contiguous().view(1,-1).repeat(_num_batch,1)

         x_t_flat = x_t_flat.view(-1)
         y_t_flat = y_t_flat.view(-1)

         d_t_flat = d_offset.contiguous().view(-1)

         one = torch.ones_like(x_t_flat)
         xyone = torch.cat((x_t_flat.unsqueeze(0),y_t_flat.unsqueeze(0),one.unsqueeze(0)),0)
         
         lonlatone = torch.matmul(Pinv,xyone)
         lon = lonlatone[0]
         lat = lonlatone[1]
               
         nt = torch.tan(lon)
         ny = torch.sin(lat)
         nyny = ny * ny

         nx = torch.sqrt((1-ny*ny)/(1 + nt*nt)) * nt
         nz = torch.sqrt(1 - nx*nx - ny*ny)

         nx_add = d_t_flat * _width_f / angular_precision
         nx = nx + nx_add
    
         lon = torch.atan(nx/nz)
         lat = torch.asin(ny/torch.sqrt(nx*nx + ny*ny + nz * nz))

         lonlatone = torch.cat((lon.unsqueeze(0),lat.unsqueeze(0),torch.ones_like(lon).unsqueeze(0)),0).cuda()
         xyone = torch.tensordot(P,lonlatone,1)

         x_t_flat = xyone[0]
         y_t_flat = xyone[1]

         input_transformed = _interpolate(input_images,x_t_flat,y_t_flat)

         output = input_transformed.view(_num_batch,_height,_width,_num_channels)
         
         return output
    output = _transform(input_images, d_offset)
 
    return output
