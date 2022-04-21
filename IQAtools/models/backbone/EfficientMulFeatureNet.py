import torch
from torch import nn
import timm
from ..builder import Model_List

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


def deconvs_upsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def deconvs(in_channels, out_channels,kernel_size=2,stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def convs(in_channels, out_channels, padding=0,kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


@Model_List.insert_Module()
class EfficientB2MulFeatureWeightNetMall(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB2MulFeatureWeightNetMall, self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(88,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(120,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.conv5_5 = convs(64,1,0,1)
        
        #第六层
        self.conv6_1 = convs(208,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(6,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)

        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(concat_feature_again)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        
        return final_score

@Model_List.insert_Module()
class EfficientB2MulFeatureWeightNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB2MulFeatureWeightNet, self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
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
                model.blocks[4]
                )
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)
        #第四层
        self.conv4_1 = convs(120,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #分数特征汇聚层
        self.conv5_1 = convs(4,32,0,1)
        self.conv5_2 = convs(32,16,1,3)
        self.conv5_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv6_1 = convs(4,32,0,1)
        self.conv6_2 = convs(32,16,1,3)
        self.conv6_3 = convs(16,1,1,3)
        
        #分数
        self.final_score = nn.AdaptiveAvgPool2d(1)
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)
        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,
                                          concat_feature_3,concat_feature_4],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv5_1(concat_feature_again)
        score_feature = self.conv5_2(score_feature)
        score_feature = self.conv5_3(score_feature)
        
        weight_feature = self.conv6_1(concat_feature_again)
        weight_feature = self.conv6_2(weight_feature)
        weight_feature = self.conv6_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        #weight_feature = torch.sigmoid(weight_feature)
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))       
        return final_score
    

@Model_List.insert_Module()
class EfficientB2MulFeatureNetMall(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB2MulFeatureNetMall, self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(88,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(120,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.conv5_5 = convs(64,1,0,1)
        
        #第五层
        self.conv6_1 = convs(208,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        #分数
        self.final_score = nn.AdaptiveAvgPool2d(1)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)

        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)

        final_score = self.final_score(score_feature)
        final_score = final_score.squeeze()
        
        return final_score


@Model_List.insert_Module()
class EfficientB2MulFeatureNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB2MulFeatureNet, self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
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
                model.blocks[4]
                )
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)
        #第四层
        self.conv4_1 = convs(120,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #分数特征汇聚层
        self.conv5_1 = convs(4,32,0,1)
        self.conv5_2 = convs(32,16,1,3)
        self.conv5_3 = convs(16,1,1,3)    
        #分数
        self.final_score = nn.AdaptiveAvgPool2d(1)
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)
        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,
                                          concat_feature_3,concat_feature_4],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv5_1(concat_feature_again)
        score_feature = self.conv5_2(score_feature)
        score_feature = self.conv5_3(score_feature)
        
        final_score = self.final_score(score_feature)
        final_score = final_score.squeeze()     
        return final_score
    
    
@Model_List.insert_Module()
class EfficientB3MulFeatureWeightNetMall(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB3MulFeatureWeightNetMall, self).__init__()
        model = timm.create_model('efficientnet_b3', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(24,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(32,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(96,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(136,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.conv5_5 = convs(64,1,0,1)
        
        #第五层
        self.conv6_1 = convs(232,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(6,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)

        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(concat_feature_again)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        
        return final_score
    




@Model_List.insert_Module()
class EfficientB1MulFeatureWeightNetMall(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB1MulFeatureWeightNetMall, self).__init__()
        model = timm.create_model('efficientnet_b1', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(40,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(80,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(112,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.conv5_5 = convs(64,1,0,1)
        
        #第五层
        self.conv6_1 = convs(192,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(6,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)
        #for feature in dis_feature:
        #    print(feature.shape)
        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(concat_feature_again)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        
        return final_score




@Model_List.insert_Module()
class EfficientB0MulFeatureWeightNetMall(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB0MulFeatureWeightNetMall, self).__init__()
        model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(40,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(80,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(112,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.conv5_5 = convs(64,1,0,1)
        
        #第五层
        self.conv6_1 = convs(192,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(6,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)
        #for feature in dis_feature:
        #    print(feature.shape)
        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(concat_feature_again)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        
        return final_score






from .cbam import CBAM


@Model_List.insert_Module()
class EfficientB2MulFeatureWeightNetMallCBAM(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB2MulFeatureWeightNetMallCBAM, self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.CBAM1 = CBAM(16,4)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.CBAM2 = CBAM(24,4)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.CBAM3 = CBAM(48,6)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(88,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.CBAM4 = CBAM(64,8)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(120,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.CBAM5 = CBAM(64,8)
        self.conv5_5 = convs(64,1,0,1)
        
        #第六层
        self.conv6_1 = convs(208,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.CBAM6 = CBAM(64,8)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(6,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)

        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.CBAM1(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.CBAM2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.CBAM3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.CBAM4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.CBAM5(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.CBAM6(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(concat_feature_again)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        
        return final_score


@Model_List.insert_Module()
class EfficientB2MulFeatureWeightNetMallX(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB2MulFeatureWeightNetMallX, self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(88,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(120,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.conv5_5 = convs(64,1,0,1)
        
        #第六层
        self.conv6_1 = convs(208,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(6,32,0,1)
        self.conv8_2 = convs(32,16,15,31)
        self.conv8_3 = convs(16,1,1,3)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)

        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(concat_feature_again)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        return final_score


@Model_List.insert_Module()
class EfficientB2MulFeatureWeightNetMall_addstd(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientB2MulFeatureWeightNetMall_addstd, self).__init__()
        model = timm.create_model('efficientnet_b2', pretrained=pretrained)
        feature_extractor_blocks = [
            nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.act1,
                model.blocks[0],
            ),
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4],
            model.blocks[5],
            ]
        self.Extractor = EfficientNetFeatureExtractor(feature_extractor_blocks)
        #第一层
        self.conv1_1 = convs(16,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(24,24,1,3)
        self.conv2_2 = deconvs(24,24)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(48,48,1,3)
        self.conv3_2 = deconvs(48,48)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,0,1)       
        
        #第四层
        self.conv4_1 = convs(88,64,1,3)
        self.conv4_2 = deconvs(64,64)
        self.conv4_3 = deconvs(64,64)
        self.conv4_4 = deconvs(64,64)
        self.conv4_5 = convs(64,1,0,1)
        
        #第五层
        self.conv5_1 = convs(120,64,1,3)
        self.conv5_2 = deconvs(64,64)
        self.conv5_3 = deconvs(64,64)
        self.conv5_4 = deconvs(64,64)
        self.conv5_5 = convs(64,1,0,1)
        
        #第六层
        self.conv6_1 = convs(208,128,1,3)
        self.conv6_2 = deconvs(128,64)
        self.conv6_3 = deconvs(64,64)
        self.conv6_4 = deconvs(64,64)
        self.conv6_5 = deconvs(64,64)
        self.conv6_6 = convs(64,1,0,1)
        
        
        #分数特征汇聚层
        self.conv7_1 = convs(6,32,0,1)
        self.conv7_2 = convs(32,16,1,3)
        self.conv7_3 = convs(16,1,1,3)
        
        #权重特征汇聚层
        self.conv8_1 = convs(6,32,0,1)
        self.conv8_2 = convs(32,16,1,3)
        self.conv8_3 = convs(16,1,1,3)
        
    def _center_crop(self, skip, x):

        _, _, h1, w1 = skip.shape
        _, _, h2, w2 = x.shape
        ht, wt = min(h1, h2), min(w1, w2)
        dh1 = (h1 - ht) // 2 if h1 > ht else 0
        dw1 = (w1 - wt) // 2 if w1 > wt else 0
        dh2 = (h2 - ht) // 2 if h2 > ht else 0
        dw2 = (w2 - wt) // 2 if w2 > wt else 0
        return skip[:, :, dh1: (dh1 + ht), dw1: (dw1 + wt)], \
                x[:, :, dh2: (dh2 + ht), dw2: (dw2 + wt)]

    def forward(self,dis):
        dis_feature = self.Extractor(dis)

        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.conv1_1(concat_feature_1)
        concat_feature_1 = self.conv1_2(concat_feature_1)
        concat_feature_1 = self.conv1_3(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.conv2_1(concat_feature_2)
        concat_feature_2 = self.conv2_2(concat_feature_2)
        concat_feature_2 = self.conv2_3(concat_feature_2)
        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.conv3_1(concat_feature_3)
        concat_feature_3 = self.conv3_2(concat_feature_3)
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_2(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_2(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_2(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
        concat_feature_6 = self.conv6_6(concat_feature_6)
        #print(concat_feature_6.shape)
        
        concat_feature_1,concat_feature_2 = self._center_crop(concat_feature_1,concat_feature_2)
        concat_feature_1,concat_feature_3 = self._center_crop(concat_feature_1,concat_feature_3)
        concat_feature_1,concat_feature_4 = self._center_crop(concat_feature_1,concat_feature_4)
        concat_feature_1,concat_feature_5 = self._center_crop(concat_feature_1,concat_feature_5)
        concat_feature_1,concat_feature_6 = self._center_crop(concat_feature_1,concat_feature_6)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,concat_feature_3,
                                          concat_feature_4,concat_feature_5,concat_feature_6],1)
        #print(concat_feature_again.shape)
        score_feature = self.conv7_1(concat_feature_again)
        score_feature = self.conv7_2(score_feature)
        score_feature = self.conv7_3(score_feature)
        
        weight_feature = self.conv8_1(concat_feature_again)
        weight_feature = self.conv8_2(weight_feature)
        weight_feature = self.conv8_3(weight_feature)      
        
        weight_feature_tmp = torch.clamp(weight_feature,min=0) + 1e-8
        weight_feature = weight_feature_tmp / torch.sum(weight_feature_tmp,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        final_score = final_score.unsqueeze(1)
        
        return final_score,score_feature,weight_feature_tmp