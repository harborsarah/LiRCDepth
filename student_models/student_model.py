import torch
import torch.nn as nn
from student_models.decoder import decoder_student
from student_models.image_model import encoder_image_student
from student_models.radar_model import encoder_radar_student
from student_models.modules import ConfModule, ConfModuleRad

class LiRCDepth(nn.Module):
    def __init__(self, params):
        super(LiRCDepth, self).__init__()
        self.params = params
        if params.radar_confidence:
            # self.rad_conf = ConfModule(params)
            self.rad_conf = ConfModuleRad(params)
            
        self.encoder = encoder_image_student(params)
        self.encoder_radar = encoder_radar_student(params)
        self.decoder = decoder_student(params, self.encoder.feat_out_channels, self.encoder_radar.feat_out_channels)

    def forward(self, x, radar):
        radar_confidence = None
        if self.params.radar_confidence:
            # radar_confidence = self.rad_conf(x, radar)
            radar_confidence = self.rad_conf(radar)

        skip_feat = self.encoder(x)
        skip_feat_radar = self.encoder_radar(radar)
        feat, depth, final_depth = self.decoder(skip_feat, skip_feat_radar, radar_confidence)
        
        if self.params.radar_confidence:
            return feat, skip_feat_radar, depth, final_depth, radar_confidence
        
        return feat, skip_feat, skip_feat_radar, depth, final_depth
        