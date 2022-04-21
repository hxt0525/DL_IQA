from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn
import torchvision
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from .unetplusplus.decoder import UnetPlusPlusDecoder
from .unet.decoder import UnetDecoder
from .fpn.decoder import FPNDecoder
from ..builder import Model_List
import timm
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, blocks):
        super(EfficientNetFeatureExtractor, self).__init__()
        assert len(blocks) > 0
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        Features = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            Features.append(x)
        return Features # 只能以这种方式返回多个tensor

def convs(in_channels, out_channels, padding=0,kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    

@Model_List.insert_Module()
class EffB2_UNetPlusPlusPrune(nn.Module):
    def __init__(self,in_ch=3,pretrained=True):
        super(EffB2_UNetPlusPlusPrune,self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(),
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            nn.Sequential(
                model.blocks[3],
                model.blocks[4],
            ),
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks) 
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=(3,16,24,48,120,208),
            decoder_channels=(128,64,32,16,8),
            n_blocks= 5,
            use_batchnorm=True,
            center= False,
            attention_type=None,
        )
        #分数特征汇聚层
        self.conv7_1 = convs(8,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(8,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def forward(self,x):
        features = self.Extractor(x)
        #for xx in features:
        #    print(xx.shape)
        decoder_output = self.decoder(*features)
        
        score_feature = self.conv7_1(decoder_output)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(decoder_output)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        return final_score



@Model_List.insert_Module()
class EffB2_UNetPrune(nn.Module):
    def __init__(self,in_ch=3,pretrained=True):
        super(EffB2_UNetPrune,self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(),
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            nn.Sequential(
                model.blocks[3],
                model.blocks[4],
            ),
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks) 
        self.decoder = UnetDecoder(
            encoder_channels=(3,16,24,48,120,208),
            decoder_channels=(128,64,32,16,8),
            n_blocks= 5,
            use_batchnorm=True,
            center= False,
            attention_type=None,
        )
        #分数特征汇聚层
        self.conv7_1 = convs(8,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(8,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def forward(self,x):
        features = self.Extractor(x)
        #for xx in features:
        #    print(xx.shape)
        decoder_output = self.decoder(*features)
        
        score_feature = self.conv7_1(decoder_output)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(decoder_output)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        return final_score
    


@Model_List.insert_Module()
class EffB2_FPNPrune(nn.Module):
    def __init__(self,in_ch=3,pretrained=True):
        super(EffB2_FPNPrune,self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(),
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            nn.Sequential(
                model.blocks[3],
                model.blocks[4],
            ),
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks) 
        self.decoder = FPNDecoder(
            encoder_channels=(3,16,24,48,120,208),
            encoder_depth=5,
            pyramid_channels=128,
            segmentation_channels=64,
            dropout=0.2,
            merge_policy='add',
        )
        #分数特征汇聚层
        self.conv7_1 = convs(64,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(64,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def forward(self,x):
        features = self.Extractor(x)
        #for xx in features:
        #    print(xx.shape)
        decoder_output = self.decoder(*features)
        
        score_feature = self.conv7_1(decoder_output)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(decoder_output)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        return final_score
    
    


@Model_List.insert_Module()
class EffB2_UNetRestore(nn.Module):
    def __init__(self,in_ch=3,pretrained=True):
        super(EffB2_UNetRestore,self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(),
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            nn.Sequential(
                model.blocks[3],
                model.blocks[4],
            ),
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks) 
        self.decoder = UnetDecoder(
            encoder_channels=(3,16,24,48,120,208),
            decoder_channels=(128,64,32,16,8),
            n_blocks= 5,
            use_batchnorm=True,
            center= False,
            attention_type=None,
        )
        #分数特征汇聚层
        self.restore = convs(8,3,0,1)

        
    def forward(self,x):
        features = self.Extractor(x)
        decoder_output = self.decoder(*features)
        
        res = self.restore(decoder_output)
        return res