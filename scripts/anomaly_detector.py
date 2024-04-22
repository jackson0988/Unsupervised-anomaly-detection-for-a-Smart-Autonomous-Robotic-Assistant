import numpy as np
import os
import cv2
import torch
from PIL import Image, ImageEnhance
#from torch import mode
from models import Encoder_Decoder_128


class AnomalyDetector(object):

    def __init__(self, base_dir, cfg):
        self.cfg = cfg
        self.use_gpu = cfg.TEST.gpu
        self.model = None
        self.model_weights = None
        self.load_model(os.path.join(base_dir, cfg.DIR.saved_models, cfg.TEST.enc_dec_model))
        self.fcsv = open(os.path.join(base_dir, "reconsloss.csv"), "w")
        self.img_win_id = cfg.PARAMS.img_win_id
        self.rec_win_id = cfg.PARAMS.rec_win_id
        self.plt_win_id = cfg.PARAMS.plt_win_id
        self.plt_step = 0
        
    def load_model(self, model_weights):
        #Insert an assertion
        self.model_weights = model_weights
        self.model = Encoder_Decoder_128()
        state_dict = torch.load(self.model_weights)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        if self.use_gpu:
            self.model = torch.nn.DataParallel(self.model).cuda()
   
    def getAnomalyScore(self, img):
        # resize input image to 128,128        
        img = cv2.resize(img, (128, 128))
        # img = transforms.Resize((128, 128), Image.BILINEAR)(img)
        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        # Image brightess enhancer
        enhancer = ImageEnhance.Brightness(pil_image)
        factor = 2.5 #adjust brightness of the image
        im_output = enhancer.enhance(factor)
        open_cv_image = np.array(im_output)
        # Convert RGB to BGR 
        img = open_cv_image[:, :, ::-1].copy()
        img = torch.from_numpy(np.asarray(img))
        # self.vis.image(img.permute(2, 0, 1), win=self.img_win_id) 
        if self.use_gpu:
            img = img.float().cuda()  
        # expand to four dimensions
        img = (img / 255 - 0.5) * 2
        img = img.permute(2, 0, 1)                         
        img = torch.unsqueeze(img, 0)
        self.model.eval()
        with torch.no_grad():
            reconstructed_img = self.model(img)
            # self.vis.image(reconstructed_img[0], win=self.rec_win_id)
            reconstructed_img = (reconstructed_img / 2 + 0.5 * 255)
            img = (img / 2 + 0.5) * 255
            ano_score = torch.mean(torch.abs(reconstructed_img-img),dim = [1,2,3])
            self.fcsv.write("{}\t{}\n".format(self.plt_step, ano_score.item()))
            self.plt_step += 1	
        return ano_score
