import torch
import os
import math
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# from Attention import Self_Attn
from .Attention_ver4 import Attn_conv
###############################################################################
# Functions
###############################################################################

def pad_tensor(input):
	height_org, width_org = input.shape[2], input.shape[3]
	divide = 16

	if width_org % divide != 0 or height_org % divide != 0:
		width_res = width_org % divide
		height_res = height_org % divide
		if width_res != 0:
			width_div = divide - width_res
			pad_left = int(width_div/2)
			pad_right = int(width_div - pad_left)
		else:
			pad_left = 0
			pad_right = 0

		if height_res != 0:
			height_div = divide - height_res
			pad_top = int(height_div/2)
			pad_bottom = int(height_div-pad_top)
		else:
			pad_top = 0
			pad_bottom = 0

		padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
		input = padding(input)
	else :
		pad_left = 0
		pad_right = 0
		pad_top = 0
		pad_bottom = 0

	height, width = input.data.shape[2], input.data.shape[3]
	assert width % divide == 0, 'width cant divided by stride'
	assert height % divide == 0, 'height cant divided by stride'
	return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
	height, width = input.shape[2], input.shape[3]
	return input[:,:,pad_top: height - pad_bottom, pad_left: width - pad_right]

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
	if norm_type =='batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	else :
		raise NotImplementedError('normalizaition layer [%s] is not found' % norm)
	return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False, opt=None):
	netG = None
	use_gpu = len(gpu_ids) > 0
	norm_layer = get_norm_layer(norm_type=norm)
	if use_gpu:
		assert(torch.cuda.is_available())
	netG = Unet_resize_conv(opt, skip)
	if len(gpu_ids) >=0 :
		netG.cuda(device=gpu_ids[0])
		netG = torch.nn.DataParallel(netG, gpu_ids)
	netG.apply(weights_init)
	return netG

def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
	netD = None
	use_gpu = len(gpu_ids) > 0
	norm_layer = get_norm_layer(norm_type=norm)
	netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid = use_sigmoid, gpu_ids=gpu_ids)
	if use_gpu:
		netD.cuda(device=gpu_ids[0])
		netD = torch.nn.DataParallel(netD, gpu_ids)
	netD.apply(weights_init)
	return netD

def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class NoNormDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
		super(NoNormDiscriminator, self).__init__()
		self.gpu_ids = gpu_ids

		kw = 4
		padw = int(np.ceil((kw-1)/2))
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mutl = min(2**n, 8)
			sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult_prev = nf_mult
		sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

		if use_sigmoid:
			sequence += [nn.Sigmoid()]

		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		return self.model(input)



class Unet_resize_conv(nn.Module):
	def __init__(self, opt, skip):
		super(Unet_resize_conv, self).__init__()
		self.opt = opt
		self.skip = skip
		p = 1

		self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
		self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn1_1 = nn.BatchNorm2d(32)

		self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
		self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn1_2 = nn.BatchNorm2d(32)
		self.max_pool1 = nn.MaxPool2d(2)

		self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
		self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn2_1 = nn.BatchNorm2d(64)

		self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
		self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn2_2 = nn.BatchNorm2d(64)
		self.max_pool2 = nn.MaxPool2d(2)

		self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
		self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn3_1 = nn.BatchNorm2d(128)

		self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
		self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn3_2 = nn.BatchNorm2d(128)
		self.max_pool3 = nn.MaxPool2d(2)

		self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
		self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn4_1 = nn.BatchNorm2d(256)

		self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
		self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn4_2 = nn.BatchNorm2d(256)
		self.max_pool4 = nn.MaxPool2d(2)

		self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
		self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn5_1 = nn.BatchNorm2d(512)

		self.conv5_2 = nn.Conv2d(512, 512 ,3, padding=p)
		self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn5_2 = nn.BatchNorm2d(512)

		self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
		self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
		self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn6_1 = nn.BatchNorm2d(256)

		self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
		self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn6_2 = nn.BatchNorm2d(256)

		self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
		self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
		self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn7_1 = nn.BatchNorm2d(128)

		self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
		self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn7_2 = nn.BatchNorm2d(128)

		self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
		self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
		self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn8_1 = nn.BatchNorm2d(64)

		self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
		self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
		self.bn8_2 = nn.BatchNorm2d(64)

		self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
		self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
		self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
		self.bn9_1 = nn.BatchNorm2d(32)

		self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
		self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv10 = nn.Conv2d(32, 3, 1)

		self.attn_conv = Attn_conv()
		self.deconv_attn = nn.Conv2d(1024,512, 3, padding=p)

	def depth_to_space(self, input, block_size):
		block_size_sq = block_size*block_size
		output = input.permute(0, 2, 3, 1)
		(batch_size, d_height, d_width, d_depth) = output.size()
		s_depth = int(d_depth/block_size_sq)
		s_width = int(d_width * block_size)
		s_height = int(d_height * block_size)
		t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
		spl = t_1.split(block_size, 3)
		stcak = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
		output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).resize(batch_size, s_height, s_width, s_depth)
		output = output.permute(0, 3, 1, 2)
		return output

	def forward(self, input, gray):
		flag = 0
		if input.size()[3] > 2200:
			avg = nn.AvgPool2d(2)
			input = avg(input)
			flag = 1
		input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)

		in_attn_conv = self.attn_conv(input) # 512,14,14

		x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
		conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
		x = self.max_pool1(conv1)

		x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
		conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
		x = self.max_pool2(conv2)

		x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
		conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
		x = self.max_pool3(conv3)

		x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
		conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
		x = self.max_pool4(conv4)

		x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
		conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

		conv5_attn_in = torch.cat([conv5,in_attn_conv], 1) # <1,1024,14,14>
		attn_deconv = self.deconv_attn(conv5_attn_in) # <1,512,14,14>

		attn_up = F.upsample(attn_deconv, scale_factor=2, mode='bilinear')
		
		# conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear') # <1,512,28,28>
		up6 = torch.cat([self.deconv5(attn_up), conv4], 1)
		x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
		conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x))) # <1,256,28,28>

		conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
		up7 = torch.cat([self.deconv6(conv6), conv3], 1)
		x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
		conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

		conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
		up8 = torch.cat([self.deconv7(conv7), conv2], 1)
		x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
		conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

		conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
		up9 = torch.cat([self.deconv8(conv8), conv1], 1)
		x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
		conv9 = self.LReLU9_2(self.conv9_2(x))


		latent = self.conv10(conv9)

		output = latent
		output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
		latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
		if flag == 1:
			output = F.upsample(output, scale_factor=2, mode='bilinear')
		return output, latent
            
def vgg_preprocess(batch, opt):
	tensortype = type(batch.data)
	(r,g,b) = torch.chunk(batch, 3, dim = 1)
	batch = torch.cat((b,g,r), dim = 1)
	batch = (batch + 1)*255*0.5
	if opt.vgg_mean:
		mean = tensortype(batch.data.size())
		mean[:,0,:,:] = 103.939
		mean[:,1,:,:] = 116.779
		mean[:,2,:,:] = 123.680
		batch = batch.sub(Variable(mean))
	return batch

class PerceptualLoss(nn.Module):
	def __init__(self, opt):
		super(PerceptualLoss, self).__init__()
		self.opt = opt
		self.instancenorm = nn.InstanceNorm2d(512, affine=False)
	def compute_vgg_loss(self, vgg, img, target):
		img_vgg = vgg_preprocess(img, self.opt)
		target_vgg = vgg_preprocess(target, self.opt)
		img_fea = vgg(img_vgg, self.opt)
		target_fea = vgg(target_vgg, self.opt)
		return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

def load_vgg16(model_dir, gpu_ids):
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	vgg = Vgg16()
	vgg.cuda(device=gpu_ids[0])
	vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
	vgg = torch.nn.DataParallel(vgg, gpu_ids)
	return vgg


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2) 
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3
 