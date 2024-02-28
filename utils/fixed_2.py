
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
# from net import NestFuse_light2_nodense, Fusion_network
#from change_22_net_1_mini import NestFuse_light2_nodense, Fusion_network, Decoder, Encoder
from args_fusion import args
import pytorch_msssim
import matplotlib.pyplot as plt

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = 'three'

import numpy as np
from models import *

import torch
import torch.optim
from skimage import img_as_ubyte

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils2.denoising_utils import *

import cv2
from models2 import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

EPSILON = 1e-5

import random
seed = 22  # 20 for 0.00005,  随机0.0001,    0.0002,   0.0005
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    # # original_imgs_path, _ = utils2.list_images(args.dataset_ir)
    # train_num = two
    # original_imgs_path = original_imgs_path[:train_num]
    # random.shuffle(original_imgs_path)
    original_imgs_path = 'a'
    # True - RGB , False - gray
    img_flag = False
    alpha_list = [700]
    w_all_list = [[6.0, 3.0]]

    for w_w in w_all_list:
        w1, w2 = w_w
        for alpha in alpha_list:
            train(original_imgs_path, img_flag, alpha, w1, w2)


def train(original_imgs_path, img_flag, alpha, w1, w2):
    batch_size = args.batch_size
    gpu=0
    # load network model
    # nc = 1
    # input_nc = nc
    # output_nc = nc
    # # nb_filter = [64, 112, 160, 208, 256]
    # nb_filter = [16, 32, 48, 64]
    # # nb_filter = [32, 64, 128, 208]
    # # nb_filter = [32, 64, 96, 128]
    # f_type = 'res'
    ################## Using GPU when it is available ##################
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    print(torch.cuda.is_available())
    print('device info:')
    print(device)
    print('#_______________________')

    ############################ Settings from noise & network
    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50

    input_depth = 1

    net = get_net(input_depth, 'skip', pad,
                         skip_n33d=128,
                         skip_n33u=128,
                         skip_n11=4,
                         num_scales=5,
                         upsample_mode='bilinear')

    net.to(device)
    # print(skip_model)


    #
    # # creating save path
    # temp_path_model = os.path.join(args.save_fusion_model)
    # temp_path_loss = os.path.join(args.save_loss_dir)
    # if os.path.exists(temp_path_model) is False:
    #     os.mkdir(temp_path_model)
    #
    # if os.path.exists(temp_path_loss) is False:
    #     os.mkdir(temp_path_loss)
    #
    # temp_path_model_w = os.path.join(args.save_fusion_model, str(w1))
    # temp_path_loss_w = os.path.join(args.save_loss_dir, str(w1))
    # if os.path.exists(temp_path_model_w) is False:
    #     os.mkdir(temp_path_model_w)
    #
    # if os.path.exists(temp_path_loss_w) is False:
    #     os.mkdir(temp_path_loss_w)



    Loss_feature = []
    Loss_ssim = []
    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_fea_loss = 0.

    # total_loss = []

    # ir代表dagl的预处理图像，vis代表restormer的预处理图像


    img_names = range(1, 13)
    # img_names = [5]
    # img_names =[11,12,two,three,four,fiveth,6,7,8,9,]
    # img_names = [two]
    # num_iter = 3000
    num_iter = 1
    psrn_maxs = []
    epoch_list = []

    # for lr in [0.00008,0.0001,0.00005,0.00002,0.0002,0.0003]:  # for LR in [0.0005, 0.001, 0.0002]:
    for lr in [0.00005]: # 0.0002,0.00050.0001,0.0002,
        #for noise in [1.5, 0.8, 0.2, 0.1]:
        for noise in [0.6]: # for noise in [1, 0.8, 0.5, 0.2, 0.1]:  # [1.5, 1, 0.8, 0.5, 0.2, 0.1]
            times = []
            # for k in range(1, 2):
            for sigma in [25]: # [15,25,50]
                #times = []
                for img_name in img_names:

                    with torch.no_grad():

                        net.load_state_dict(torch.load(r"C:\Users\TK\Desktop\image-dip-pretrain-main\models\train_denoise_pretrain_skip\netepoch_800.pth"))
                        # Compute number of parameters
                        s = sum([np.prod(list(p.size())) for p in net.parameters()]);
                        print('Number of params: %d' % s)

                        for name, param in net.named_parameters():
                            if 'up' in name:
                                param.requires_grad = False

                        net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                        print('After freezing, the left trainable param is {}'.format(net_total_params))  # Loss

                        # deepsupervision = False
                        # nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
                        #
                        # # args.resume_nestfuse = r"models/train_denoise/fusionnet/supervised_restormer/new_0_001/conv10_2/223_4/nest_modelepoch_2.model"
                        # # model_path = args.resume_nestfuse
                        # # encoder_path = "models/train_denoise/fusionnet/supervised_restormer/new_0_001/conv10_2/223_44_encoder_moer_para/nest_model_encoder_epoch_200.model"
                        # # decoder_path = "models/train_denoise/fusionnet/supervised_restormer/new_0_001/conv10_2/223_44_encoder_moer_para/nest_model_decoder_epoch_200.model"
                        # # encoder_path = "models/train_denoise/fusionnet/supervised_restormer/new_0_001/conv10_2/223_44/nest_model_encoder_epoch_1960.model"
                        # # decoder_path = "models/train_denoise/fusionnet/supervised_restormer/new_0_001/conv10_2/223_44/nest_model_decoder_epoch_1960.model"
                        #
                        # encoder_path = "models/train_denoise/fusionnet/supervised_dncnn_epoch_1200/nest_model_encoder_epoch_1200.model"
                        # decoder_path = "models/train_denoise/fusionnet/supervised_dncnn_epoch_1200/nest_model_decoder_epoch_1200.model"
                        #
                        # # load auto-encoder network
                        # print('Resuming, initializing encoder using weight from {}.'.format(encoder_path))
                        # nest_model.encoder.load_state_dict(torch.load(encoder_path))
                        # print('Resuming, initializing encoder using weight from {}.'.format(decoder_path))
                        # nest_model.decoder.load_state_dict(torch.load(decoder_path))

                        # nest_model.eval()




                    # deepsupervision = False
                    # nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
                    # decode = Decoder(nb_filter, input_nc, output_nc, deepsupervision)
                    # encode = Encoder(nb_filter, input_nc, output_nc, deepsupervision)

                    # nest_model.train()


                    # fusion network
                    # fusion_model = Fusion_network(nb_filter, f_type)
                    # args.resume_fusion_model = r"E:\zhouchangfei\denoise\imagefusion-rfn-nest-main-denoising\models\train_denoise\fusionnet\supervised_restormer\new_0_001\conv10_2\22_2\epoch_1960.model"
                    # if args.resume_fusion_model is not None:
                    #     print(
                    #         'Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
                    #     fusion_model.load_state_dict(torch.load(args.resume_fusion_model))
                    #     fusion_model.train()
                    # optimizer = Adam(fusion_model.parameters(), args.lr)
                    mse = torch.nn.MSELoss()

                    if args.cuda:
                        net.cuda()

                        # fusion_model.cuda()

                    if img_name < 10:
                        img_name = "0" + str(img_name)

                    print("the image is set12_%s" % img_name)

                    imsize = -1
                    PLOT = True
                    # sigma = 25
                    sigma_ = sigma / 255.
                    # workplace = 'result/rfn_nest_lr_change_2 _new_encoder_256/set12_sig{}/{}'.format(sigma, img_name)

                    # workplace = 'result_paper/change_22_mini_2_train_denoise_five_restormer_change_{}_ite_3000_noise_{}/{}/set12_sig{}/{}'.format(
                    #     lr, noise, k, sigma,
                    #     img_name)

                    # workplace = 'result_paper/不同算法消融_1/dncnn/set12/change_22_mini_2__{}_ite_3000_noise_{}/set12_sig{}/{}'.format(
                    #     lr, noise, sigma,
                    #     img_name)

                    workplace = 'result_paper/不同算法消融/dncnn/set12/change_22_mini_2__{}_ite_3000_noise_{}/set12_sig{}/{}'.format(
                        lr, noise,  sigma,
                        img_name)
                    if not os.path.exists(workplace):
                        os.makedirs(workplace)
                    INPUT = 'noise'  # 'meshgrid'
                    pad = 'reflection'
                    # OPT_OVER = 'net'  # 'net,input'
                    OPT_OVER = 'fusion_model'
                    reg_noise_std = 1. / 30.  # set to one./20. for sigma=50
                    # LR = 0.0001 * fiveth  # LR = 0.0001*fiveth
                    # LR = 0.00001 # [four,fiveth,11]
                    # LR = 0.0001 * fiveth
                    # LR = 0.0001 * three
                    # LR = 0.0001 * 5

                    OPTIMIZER = 'adam'  # 'LBFGS'
                    show_every = 200
                    exp_weight = 0.99

                    input_depth = 2
                    figsize = 4
                    # deJPEG
                    # fname = 'data/denoising/snail.jpg'

                    out_avg = None
                    last_net = None
                    psrn_noisy_last = 0
                    i = 0
                    psrn_max = 0
                    ## denoising
                    fname = 'data/denoising/set12/{}.png'.format(img_name)

                    # Add synthetic noise
                    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
                    img_np = pil_to_np(img_pil)
                    # img_np = img_np * 255.0

                    img_torch = np_to_torch(img_np).type(dtype)
                    img_noisy_np = np.load('data/denoising/set12_sig{}_noisy/{}.npy'.format(sigma, img_name))[None, :]
                    # img_noisy_np = img_noisy_np * 255.0
                    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

                    # img_noisy_torch = img_noisy_torch *255
                    # 这里的restormer指的是不同对比算法  ffdnet
                    pre_restormer_fname = 'data/denoising/set12_dncnn_{}/{}.png'.format(sigma, img_name)
                    img_restormer_pil = crop_image(get_image(pre_restormer_fname, imsize)[0], d=32)
                    img_restormer_np = pil_to_np(img_restormer_pil)
                    img_restormer_np = img_restormer_np * 255
                    img_restormer_torch = np_to_torch(img_restormer_np).type(dtype)

                    # pre_dagl_fname = 'data/denoising/set12_sig{}_dagl/{}.png'.format(sigma,img_name)
                    # img_dagl_pil = crop_image(get_image(pre_dagl_fname, imsize)[0], d=32)
                    # img_dagl_np = pil_to_np(img_dagl_pil)
                    # img_dagl_np = np.load('data/denoising/set12_sig{}_dagl/{}.npy'.format(sigma, img_name))[None, :]
                    # img_dagl_np = img_dagl_np * 255.0
                    # img_dagl_torch = np_to_torch(img_dagl_np).type(dtype)

                    # img_ir为img_restormer,img_vi为img_dagl
                    # img_ir = img_restormer_torch
                    # img_vi = img_dagl_torch

                    # 创建一个文本文件并清空之前的记录
                    file = open(os.path.join(workplace, "logs.txt"), 'w')
                    file.truncate(0)

                    print('Starting optimization with ADAM')

                    # optimizer = torch.optim.Adam(nest_model.parameters(), lr=lr)
                    # s = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
                    # print('Number of params: %d' % s)

                    optimizer = torch.optim.Adam(nest_model.decoder.parameters(), lr=lr)
                    s = sum([np.prod(list(p.size())) for p in nest_model.decoder.parameters()])
                    print('Number of params: %d' % s)


                    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)

                    avg_fname = 'data/denoising/set12_sig{}_dagl_restormer_avg_fusion/{}.bmp'.format(sigma, img_name)
                    img_avg_pil = crop_image(get_image(avg_fname, imsize)[0], d=32)
                    img_avg_np = pil_to_np(img_avg_pil)
                    img_avg_np = img_avg_np * 255.0
                    img_avg_torch = np_to_torch(img_avg_np).type(dtype)
                    img_restormer_torch = Variable(img_restormer_torch, requires_grad=False)
                    # img_dagl_torch = Variable(img_dagl_torch, requires_grad=False)

                    if args.cuda:
                        img_restormer_torch = img_restormer_torch.cuda()
                        # img_dagl_torch = img_dagl_torch.cuda()
                    strat = time.time()
                    f = nest_model.encoder(img_restormer_torch)
                    f2 = [m.detach().clone() for m in f]
                    for j in range(num_iter):
                        optimizer.zero_grad()
                        # closure()函数用来前向传播，并计算损失
                        # 求出output
                        # img_restormer_torch = Variable(img_restormer_torch, requires_grad=False)
                        # img_dagl_torch = Variable(img_dagl_torch, requires_grad=False)
                        #
                        # if args.cuda:
                        #     img_restormer_torch = img_restormer_torch.cuda()
                        #     img_dagl_torch = img_dagl_torch.cuda()

                        # get fusion image
                        # encoder
                        # en_ir = nest_model.encoder(img_restormer_torch)
                        # # en_vi = nest_model.encoder(img_dagl_torch)
                        # # en_vi = en_ir
                        # # fusion
                        # # f = fusion_model(en_ir)
                        # # decoder
                        # out = nest_model.decoder_eval(en_ir)
                        # out = nest_model(img_restormer_torch,deepsupervision)
                        out = nest_model.decoder(f2, deepsupervision)
                        # out = out * 255.0

                        # out = torch.tensor([item.cpu().detach().numpy() for item in outputs[0]]).cuda()

                        # Smoothing
                        if out_avg is None:
                            out_avg = out.detach()
                        else:
                            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

                        # total_loss = mse(out[:,0:one,:,:], img_dagl_torch)+mse(out[:,0:one,:,:], img_restormer_torch)+mse(out[:,one:two,:,:],img_restormer_torch)+mse(out[:,one:two,:,:], img_dagl_torch)
                        total_loss = mse(out, img_restormer_torch/255) + mse(out, img_noisy_torch) * noise  # * 0.1
                        total_loss.requires_grad_(True)
                        total_loss.backward()

                        psrn = compare_psnr(img_np, out.detach().cpu().numpy()[0])
                        psrn_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

                        # Note that we do not have GT for the "snail" example
                        # # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
                        print('Iteration %05d    Loss %f   psrn: %f  psrn_sm:%f  lr:%f' % (
                            i, total_loss.item(), psrn, psrn_sm, optimizer.param_groups[0]['lr']))
                        file.write('Iteration %05d    Loss %f   psrn: %f  psrn_sm:%f  lr:%f\n' % (
                            i, total_loss.item(), psrn, psrn_sm, optimizer.param_groups[0]['lr']))

                        if psrn > psrn_max:
                            psrn_max = psrn
                            epoch_max = i + 1
                            # cv2.imwrite(os.path.join(workplace,"best_dagl.bmp"),img_as_ubyte(out_avg[:,0:one,:,:].detach().cpu().numpy()[0]))
                            pil_max = np_to_pil(out.detach().cpu().numpy()[0])
                            pil_max.save(os.path.join(workplace, "best_channel0.bmp"))
                            # pil_max.save(os.path.join(workplace, 'best_epoch'+str(i+one)+'_psrn_max_'+str(psrn_max)+'.bmp'))
                        # Backtracking
                        if i % show_every:
                            if psrn - psrn_noisy_last < -50:
                                print('Falling back to previous checkpoint.')

                                return total_loss * 0
                            else:
                                # last_net = [x.detach().cpu() for x in net.parameters()]
                                psrn_noisy_last = psrn
                        i += 1
                        optimizer.step()

                        # # scheduler.step()
                        # if (j+1) % 800 == 0 and j < 1600:
                        #     for param_group in optimizer.param_groups:
                        #         param_group['lr'] = lr * 0.5
                        #         lr = param_group['lr']


                    end = time.time()
                    times.append(end - strat)
                    # print("time: ", end - strat)
                    psrn_maxs.append(psrn_max)
                    epoch_list.append(epoch_max)

                    file.write('Iteration_max %05d    Loss %f   psrn_max: %f  psrn_sm:%f\n' % (
                        epoch_max, total_loss.item(), psrn_max, psrn_sm))
                    print('psrn_maxs: ', psrn_maxs, "   avg_max: ", np.mean(np.array(psrn_maxs)))
                    print('epoch_list: ', epoch_list, "   avg_max: ", np.mean(np.array(psrn_maxs)))
            print("times_avg: %f" % np.mean(np.array(times)))
            print(times)
            file.write("times_avg: "% (np.mean(np.array(times))))


if __name__ == "__main__":
    main()