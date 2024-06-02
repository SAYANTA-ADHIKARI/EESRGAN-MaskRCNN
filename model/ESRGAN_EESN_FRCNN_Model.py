import logging
from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
import model.model as model
import model.lr_scheduler as lr_scheduler
import kornia
from model.loss import GANLoss, CharbonnierLoss
from .gan_base_model import BaseModel
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from detection.engine import evaluate, evaluate_save
from detection.utils import reduce_dict
import numpy as np
import sys
from utils.canny import canny

logger = logging.getLogger('base')
# Taken from ESRGAN BASICSR repository and modified
class ESRGAN_EESN_FRCNN_Model(BaseModel):
    def __init__(self, config, device):
        super(ESRGAN_EESN_FRCNN_Model, self).__init__(config, device)
        self.configG = config['network_G']
        self.configD = config['network_D']
        self.configT = config['train']
        self.configO = config['optimizer']['args']
        self.configS = config['lr_scheduler']
        self.config = config
        self.device = device
        #Generator
        self.netG = model.ESRGAN_EESN(in_nc=self.configG['in_nc'], out_nc=self.configG['out_nc'],
                                    nf=self.configG['nf'], nb=self.configG['nb'], 
                                    eesn_filter=self.configG['filter'], eesn_filter_size=self.configG["filter_size"])
        #self.netG.load_state_dict(torch.load('./saved/pretrained_models_EESRGAN_FRCNN/9360_G.pth'))
        self.netG = self.netG.to(self.device)
        self.netG = DataParallel(self.netG)

        #descriminator
        self.netD = model.Discriminator_VGG_128(in_nc=self.configD['in_nc'], nf=self.configD['nf'])
        #self.netD.load_state_dict(torch.load('./saved/pretrained_models_EESRGAN_FRCNN/9360_D.pth'))
        self.netD = self.netD.to(self.device)
        self.netD = DataParallel(self.netD)

        #FRCNN_model
        # self.netFRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # num_classes = 2 # car and background
        # in_features = self.netFRCNN.roi_heads.box_predictor.cls_score.in_features
        # self.netFRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # self.netFRCNN.to(self.device)
        #self.netFRCNN.load_state_dict(torch.load('./saved/pretrained_models_EESRGAN_FRCNN/9360_FRCNN.pth'))

        self.netFRCNN=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, min_size=500)
        num_classes = 12 # 11 classes + background
        in_features = self.netFRCNN.roi_heads.box_predictor.cls_score.in_features 
        self.netFRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.netFRCNN.roi_heads.mask_predictor.conv5_mask.in_channels
        self.netFRCNN.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        self.netFRCNN.to(self.device)

        #self.netFRCNN = DataParallel(self.netFRCNN)
        self.netG.train()
        self.netD.train()
        self.netFRCNN.train()
        #print(self.configT['pixel_weight'])
        # G CharbonnierLoss for final output SR and GT HR
        self.cri_charbonnier = CharbonnierLoss().to(device)
        # G pixel loss
        if self.configT['pixel_weight'] > 0.0:
            l_pix_type = self.configT['pixel_criterion']
            if l_pix_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif l_pix_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
            self.l_pix_w = self.configT['pixel_weight']
        else:
            self.cri_pix = None

        # G feature loss
        #print(self.configT['feature_weight']+1)
        if self.configT['feature_weight'] > 0:
            l_fea_type = self.configT['feature_criterion']
            if l_fea_type == 'l1':
                self.cri_fea = nn.L1Loss().to(self.device)
            elif l_fea_type == 'l2':
                self.cri_fea = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
            self.l_fea_w = self.configT['feature_weight']
        else:
            self.cri_fea = None
        if self.cri_fea:  # load VGG perceptual loss
            self.netF = model.VGGFeatureExtractor(feature_layer=34,
                                          use_input_norm=True, device=self.device)
            self.netF = self.netF.to(self.device)
            self.netF = DataParallel(self.netF)
            self.netF.eval()

        # GD gan loss
        self.cri_gan = GANLoss(self.configT['gan_type'], 1.0, 0.0).to(self.device)
        self.l_gan_w = self.configT['gan_weight']
        # D_update_ratio and D_init_iters
        self.D_update_ratio = self.configT['D_update_ratio'] if self.configT['D_update_ratio'] else 1
        self.D_init_iters = self.configT['D_init_iters'] if self.configT['D_init_iters'] else 0


        # optimizers
        # G
        wd_G = self.configO['weight_decay_G'] if self.configO['weight_decay_G'] else 0
        optim_params = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer_G = torch.optim.Adam(optim_params, lr=self.configO['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(self.configO['beta1_G'], self.configO['beta2_G']))
        self.optimizers.append(self.optimizer_G)

        # D
        wd_D = self.configO['weight_decay_D'] if self.configO['weight_decay_D'] else 0
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.configO['lr_D'],
                                            weight_decay=wd_D,
                                            betas=(self.configO['beta1_D'], self.configO['beta2_D']))
        self.optimizers.append(self.optimizer_D)

        # FRCNN -- use weigt decay
        FRCNN_params = [p for p in self.netFRCNN.parameters() if p.requires_grad]
        self.optimizer_FRCNN = torch.optim.SGD(FRCNN_params, lr=0.005,
                                               momentum=0.9, weight_decay=0.0005)
        self.optimizers.append(self.optimizer_FRCNN)

        # schedulers
        if self.configS['type'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, self.configS['args']['lr_steps'],
                                                     restarts=self.configS['args']['restarts'],
                                                     weights=self.configS['args']['restart_weights'],
                                                     gamma=self.configS['args']['lr_gamma'],
                                                     clear_state=False))
        elif self.configS['type'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, self.configS['args']['T_period'], eta_min=self.configS['args']['eta_min'],
                        restarts=self.configS['args']['restarts'], weights=self.configS['args']['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
        print(self.configS['args']['restarts'])
        self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed
    '''
    The main repo did not use collate_fn and image read has different flags
    and also used np.ascontiguousarray()
    Might change my code if problem happens
    '''
    def feed_data(self, image, targets):
        self.var_L = image['image_lq'].to(self.device)
        self.var_H = image['image'].to(self.device)
        input_ref = image['ref'] if 'ref' in image else image['image']
        self.var_ref = input_ref.to(self.device)
        '''
        for t in targets:
            for k, v in t.items():
                print(v)
        '''
        self.targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]


    def optimize_parameters(self, step):
        #Generator
        for p in self.netG.parameters():
            p.requires_grad = True #NOTE: CHanged here for generator false
        for p in self.netD.parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        self.fake_H, self.final_SR, self.x_learned_lap_fake, _ = self.netG(self.var_L)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix: #pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea: # feature loss
                real_fea = self.netF(self.var_H).detach() #don't want to backpropagate this, need proper explanation
                fake_fea = self.netF(self.fake_H) #In netF normalize=False, check it
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

            pred_g_fake = self.netD(self.fake_H)
            if self.configT['gan_type'] == 'gan':
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.configT['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()   #NOTE:
                l_g_gan = self.l_gan_w * (
                self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan
            #EESN calculate loss
            ################### Change is here ###################
            if self.configG["filter"] == "laplacian":
                self.lap_HR = kornia.laplacian(self.var_H, self.configG["filter_size"])
            elif self.configG["filter"] == "sobel":
                self.lap_HR = kornia.sobel(self.var_H)
            elif self.configG["filter"] == "spatial_gradient":
                self.lap_HR1 = kornia.filters.spatial_gradient(self.var_H)[:,:,0,:,:]
                self.lap_HR2 = kornia.filters.spatial_gradient(self.var_H)[:,:,1,:,:]
                self.lap_HR = torch.sqrt(self.lap_HR1**2 + self.lap_HR2**2)
                self.lap_HR, _ = self.lap_HR.unbind(dim=-1)
            elif self.configG["filter"] == "canny":
                val_H = canny(self.var_H, hysteresis=False)
                self.lap_HR = torch.cat([val_H, val_H, val_H], dim=1)
            else:
                raise NotImplemented("Filter not implemented")
            # self.lap_HR = canny(self.lap_HR, use_cuda=True)
            # self.lap_HR = kornia.filters.canny(self.var_H, 2, 0.5, 0.5)
            # print("train", self.var_H.shape)
            # exit()
            # self.lar_HR = kornia.sobel()

            ## CANNY EDGE DETECTION
            # self.lap_HR1 = kornia.filters.spatial_gradient(self.var_H)[:,:,0,:,:]
            # self.lap_HR2 = kornia.filters.spatial_gradient(self.var_H)[:,:,1,:,:]
            # self.lap_HR = torch.sqrt(self.lap_HR1**2 + self.lap_HR2**2)
            # self.lap_HR, _ = self.lap_HR.unbind(dim=-1)
            # val_H = canny(self.var_H,hysteresis=False)
            # self.lap_HR = torch.cat([val_H, val_H, val_H], dim=1)
            # ################### Change is here ###################
            if self.cri_charbonnier: # charbonnier pixel loss HR and SR
                l_e_charbonnier = 5 * (self.cri_charbonnier(self.final_SR, self.var_H)
                                        + self.cri_charbonnier(self.x_learned_lap_fake, self.lap_HR))#change the weight to empirically
            l_g_total += l_e_charbonnier

            l_g_total.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.config["clip_value"])
            self.optimizer_G.step() #NOTE: Generator step stopped

        #descriminator
        for p in self.netD.parameters():
            p.requires_grad = True # NOTE: Change Here for discriminator false 

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach()) #to avoid BP to Generator
        if self.configT['gan_type'] == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.configT['gan_type'] == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2 # thinking of adding final sr d loss

        l_d_total.backward()
        nn.utils.clip_grad_norm_(self.netD.parameters(), self.config["clip_value"])
        self.optimizer_D.step() #NOTE: Discriminator step stopped

        '''
        Freeze EESRGAN
        '''
        #freeze Generator
        '''
        for p in self.netG.parameters():
            p.requires_grad = False
        '''
        for p in self.netD.parameters():
            p.requires_grad = False
        #Run FRCNN
        self.optimizer_FRCNN.zero_grad()
        self.intermediate_img = self.final_SR
        img_count = self.intermediate_img.size()[0]
        # print(self.intermediate_img.shape)
        # print(self.targets)
        # sys.exit()
        self.intermediate_img = [self.intermediate_img[i] for i in range(img_count)]
        # print(self.intermediate_img, self.targets)
        # print(len(self.targets[0]["masks"]), self.targets[0]["masks"].shape)
        loss_dict = self.netFRCNN(self.intermediate_img, self.targets)
        # print(self.targets)
        # print(loss_dict)
        # losses = sum(loss for loss in loss_dict.values()) #NOTE: old impl
        # weight loss accordingly to bring all in same scale!!!
        losses = torch.tensor(0.0, requires_grad = True, device = self.device)
        for k, val in loss_dict.items():
            losses = losses + self.configT["mask_rcnn_loss"][k] * val
        
        if torch.isnan(losses):
            print(f"Loss is Nan, {loss_dict}, Image ID: {self.targets}")
            sys.exit()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())  #NOTE: old impl
        # weight loss accordingly to bring all in same scale!!!
        loss_value = 0
        for k, val in loss_dict.items():
            loss_value += self.configT["mask_rcnn_loss"][k] * val.item()

        # loss_value = losses_reduced.item()

        losses.backward()
        nn.utils.clip_grad_norm_(self.netFRCNN.parameters(), self.config["clip_value"])
        self.optimizer_FRCNN.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()
            self.log_dict['l_e_charbonnier'] = l_e_charbonnier.item()

        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())
        self.log_dict['MaskRCNN_loss'] = loss_value
        for k, v in loss_dict_reduced.items():
            self.log_dict[f"MRCNN/{k}"] = v.item()

    def test(self, valid_data_loader, train = True, testResult = False):
        self.netG.eval()
        self.netFRCNN.eval()
        self.targets = valid_data_loader
        if testResult == False:
            with torch.no_grad():
                # print("test", self.var_L.shape)
                self.fake_H, self.final_SR, self.x_learned_lap_fake, self.x_lap = self.netG(self.var_L)
        ########### CHANGE IS HERE ############
                if self.configG["filter"] == "laplacian":
                    self.x_lap_HR = kornia.laplacian(self.var_H, self.configG["filter_size"])

                elif self.configG["filter"] == "sobel":
                    self.x_lap_HR = kornia.sobel(self.var_H)
                elif self.configG["filter"] == "spatial_gradient":
                    lap_HR1 = kornia.filters.spatial_gradient(self.var_H)[:,:,0,:,:]
                    lap_HR2 = kornia.filters.spatial_gradient(self.var_H)[:,:,1,:,:]
                    self.lap_HR = torch.sqrt(lap_HR1**2 + lap_HR2**2)
                    self.lap_HR, _ = self.lap_HR.unbind(dim=-1)
                elif self.configG["filter"] == "canny":
                    val_H = canny(self.var_H, hysteresis=False)
                    self.lap_HR = torch.cat([val_H, val_H, val_H], dim=1)
                else:
                    raise NotImplemented("Filter not implemented")
                
                img_count = self.final_SR.size()[0]
                image = [self.final_SR[i] for i in range(img_count)]
                outputs = self.netFRCNN(image)
                self.res = outputs
                # self.x_lap_HR = kornia.sobel(self.var_H)
                # self.x_lap_HR = canny(self.var_H, use_cuda=True)
                
                # self.x_lap_HR1 = kornia.filters.spatial_gradient(self.var_H)[:,:,0,:,:]
                # self.x_lap_HR2 = kornia.filters.spatial_gradient(self.var_H)[:,:,1,:,:]
                # self.x_lap_HR = torch.sqrt(self.x_lap_HR1**2 + self.x_lap_HR2**2)
                # self.x_lap_HR, _ = self.x_lap_HR.unbind(dim=-1)
                ## CANNY EDGE DETECTION
                # val_H = canny(self.var_H,hysteresis=False)
                # self.x_lap_HR = torch.cat([val_H, val_H, val_H], dim=1)
        ########### CHANGE IS HERE ############
        if train == True:
            _, _, log_dict = evaluate(self.netG, self.netFRCNN, self.targets, self.device, train)
            log_dict = {f"MRCNN/{k}": v for k, v in log_dict.items()}
            self.log_dict = log_dict

        if testResult == True:
            e, _, _ = evaluate(self.netG, self.netFRCNN, self.targets, self.device, False)
            # # for k, v in e.coco_eval.items():
            # #     print(k)
            # print(e.coco_eval['bbox'].eval['precision'].shape)
            # np.save('./required_precision.npy', e.coco_eval['bbox'].eval['precision'])
            evaluate_save(self.netG, self.netFRCNN, self.targets, self.device, self.config)
        self.netG.train()
        self.netFRCNN.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        #out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['lap_learned'] = self.x_learned_lap_fake.detach()[0].float().cpu()
        out_dict['lap_HR'] = self.x_lap_HR.detach()[0].float().cpu()
        out_dict['lap'] = self.x_lap.detach()[0].float().cpu()
        out_dict['final_SR'] = self.final_SR.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        out_dict['FRCNN'] = self.res
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        # Discriminator
        s, n = self.get_network_description(self.netD)
        if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                             self.netD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netD.__class__.__name__)

        logger.info('Network D structure: {}, with parameters: {:,d}'.format(
            net_struc_str, n))
        logger.info(s)

        if self.cri_fea:  # F, Perceptual Network
            s, n = self.get_network_description(self.netF)
            if isinstance(self.netF, nn.DataParallel) or isinstance(
                    self.netF, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                 self.netF.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netF.__class__.__name__)

            logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))
            logger.info(s)

        #FRCNN_model
        # Discriminator
        s, n = self.get_network_description(self.netFRCNN)
        if isinstance(self.netFRCNN, nn.DataParallel) or isinstance(self.netFRCNN,
                                                                DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netFRCNN.__class__.__name__,
                                             self.netFRCNN.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netFRCNN.__class__.__name__)

        logger.info('Network FRCNN structure: {}, with parameters: {:,d}'.format(
            net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.config['path']['pretrain_model_G']
        if load_path_G:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.config['path']['strict_load'])
        load_path_D = self.config['path']['pretrain_model_D']
        if load_path_D:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.config['path']['strict_load'])
        load_path_FRCNN = self.config['path']['pretrain_model_FRCNN']
        if load_path_FRCNN:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_FRCNN))
            self.load_network(load_path_FRCNN, self.netFRCNN, self.config['path']['strict_load'], rem = True)


    def save(self, iter_step, path=None):
        self.save_network(self.netG, 'G', iter_step, path)
        self.save_network(self.netD, 'D', iter_step, path)
        self.save_network(self.netFRCNN, 'FRCNN', iter_step, path)
        #self.save_network(self.netG.module.netE, 'E', iter_step)