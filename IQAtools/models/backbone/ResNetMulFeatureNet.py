import torch
import torch.nn as nn
import torchvision
from ..builder import Model_List

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, blocks):
        super(ResNetFeatureExtractor, self).__init__()
        assert len(blocks) > 0
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        Features = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            Features.append(x)
        return Features # 只能以这种方式返回多个tensor


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
import timm

@Model_List.insert_Module()
class ResnetMulFeatureWeightNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNet, self).__init__()
        model = timm.create_model('resnet50', pretrained=pretrained)
        extractor_blocks = [
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.act1,
                model.maxpool,
            ),
            model.layer1,
            model.layer2[0:2],
            model.layer2[2:],
            model.layer3[0:3],
            model.layer3[3:],
            ]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,32,0,1)
        self.conv1_2 = convs(32,32,1,3)
        self.conv1_3 = convs(32,1,0,1)
        #第二层
        self.conv2_1 = convs(256,64,0,1)
        self.conv2_2 = convs(64,64,1,3)
        self.conv2_3 = convs(64,1,0,1)
        #第三层
        self.conv3_1 = convs(512,128,0,1)
        self.conv3_3 = deconvs(128,64)
        self.conv3_4 = convs(64,1,1,3)       
        
        #第四层
        self.conv4_1 = convs(512,128,0,1)
        self.conv4_3 = deconvs(128,64)
        self.conv4_4 = convs(64,1,1,3)
        
        #第五层
        self.conv5_1 = convs(1024,128,0,1)
        self.conv5_3 = deconvs(128,128)
        self.conv5_4 = deconvs(128,64)
        self.conv5_5 = convs(64,1,1,3)
        
        #第六层
        self.conv6_1 = convs(1024,128,0,1)
        self.conv6_3 = deconvs(128,128)
        self.conv6_4 = deconvs(128,64)
        self.conv6_5 = convs(64,1,1,3)
        
        
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
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
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
class ResnetMulFeatureWeightNetMall(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNetMall, self).__init__()
        model = timm.create_model('resnet50', pretrained=pretrained)
        extractor_blocks = [
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.act1,
                model.maxpool,
            ),
            model.layer1,
            model.layer2[0:2],
            model.layer2[2:],
            model.layer3[0:3],
            model.layer3[3:],
            ]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,16,0,1)
        self.conv1_2 = convs(16,16,1,3)
        self.conv1_3 = convs(16,1,0,1)
        #第二层
        self.conv2_1 = convs(256,24,0,1)
        self.conv2_2 = convs(24,24,1,3)
        self.conv2_3 = convs(24,1,0,1)
        #第三层
        self.conv3_1 = convs(512,48,0,1)
        self.conv3_3 = deconvs(48,48)
        self.conv3_4 = convs(48,1,1,3)       
        
        #第四层
        self.conv4_1 = convs(512,128,0,1)
        self.conv4_3 = deconvs(128,64)
        self.conv4_4 = convs(64,1,1,3)
        
        #第五层
        self.conv5_1 = convs(1024,128,0,1)
        self.conv5_3 = deconvs(128,128)
        self.conv5_4 = deconvs(128,64)
        self.conv5_5 = convs(64,1,1,3)
        
        #第六层
        self.conv6_1 = convs(1024,128,0,1)
        self.conv6_3 = deconvs(128,128)
        self.conv6_4 = deconvs(128,64)
        self.conv6_5 = convs(64,1,1,3)
        
        
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
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
        concat_feature_6 = self.conv6_3(concat_feature_6)
        concat_feature_6 = self.conv6_4(concat_feature_6)
        concat_feature_6 = self.conv6_5(concat_feature_6)
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
'''
@Model_List.insert_Module()
class ResnetMulFeatureWeightNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNet, self).__init__()
        model = timm.create_model('resnet50', pretrained=True)
        extractor_blocks = [
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.act1,
                model.maxpool,
            ),
            model.layer1,
            model.layer2[0:2],
            model.layer2[2:],
            model.layer3[0:3],
            model.layer3[3:],
            ]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,32,0,1)
        self.conv1_2 = deconvs(32,32)
        self.conv1_3 = convs(32,1,0,1)
        #第二层
        self.conv2_1 = convs(256,128,0,1)
        self.conv2_2 = deconvs(128,64)
        self.conv2_3 = convs(64,1,1,3)
        #第三层
        self.conv3_1 = convs(512,128,0,1)
        self.conv3_2 = deconvs(128,128)
        self.conv3_3 = deconvs(128,64)
        self.conv3_4 = convs(64,1,1,3)       
        
        #第四层
        self.conv4_1 = convs(512,128,0,1)
        self.conv4_2 = deconvs(128,128)
        self.conv4_3 = deconvs(128,64)
        self.conv4_4 = convs(64,1,1,3)
        
        #第五层
        self.conv5_1 = convs(1024,128,0,1)
        self.conv5_2 = deconvs(128,128)
        self.conv5_3 = deconvs(128,128)
        self.conv5_4 = deconvs(128,64)
        self.conv5_5 = convs(64,1,1,3)
        
        #第六层
        self.conv6_1 = convs(1024,128,0,1)
        self.conv6_2 = deconvs(128,128)
        self.conv6_3 = deconvs(128,128)
        self.conv6_4 = deconvs(128,64)
        self.conv6_5 = convs(64,1,1,3)
        
        
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
'''
'''
@Model_List.insert_Module()
class ResnetMulFeatureWeightNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNet, self).__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=pretrained)
        extractor_blocks = [
        # conv1
        nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ),
        # conv2_x
        nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        ),
        # conv3_x
        resnet.layer2,
        # conv4_x
        resnet.layer3]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,64,1,1)
        self.conv1_2 = convs(64,64,1,3)
        self.conv1_3 = convs(64,1,0,3)
        #第二层
        self.conv2_1 = convs(256,256,0,1)
        self.conv2_2 = deconvs(256,128,2,2)
        self.conv2_3 = convs(128,1,1,3)
        #第三层
        self.conv3_1 = convs(512,512,0,1)
        self.conv3_2 = deconvs(512,256,2,2)
        self.conv3_3 = deconvs(256,128,2,2)
        self.conv3_4 = convs(128,1,1,3)
        #第四层
        self.conv4_1 = convs(1024,1024,0,1)
        self.conv4_2 = deconvs(1024,512,2,2)
        self.conv4_3 = deconvs(512,256,2,2)
        self.conv4_4 = deconvs(256,128,2,2)
        self.conv4_5 = convs(128,1,1,3)
        
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
        weight_feature = weight_feature / torch.sum(weight_feature,dim=(1,2,3),keepdim=True)

        score_map = torch.mul(score_feature,weight_feature)
        final_score = torch.sum(score_map,dim=(1,2,3))
        
        return final_score
'''

@Model_List.insert_Module()
class ResnetMulFeatureWeightNetS(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNetS, self).__init__()
        model = timm.create_model('resnet50', pretrained=pretrained)
        extractor_blocks = [
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.act1,
                model.maxpool,
            ),
            model.layer1,
            model.layer2[0:2],
            model.layer2[2:],
            model.layer3[0:2],
            model.layer3[2:4],
            ]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,32,0,1)
        self.conv1_2 = deconvs(32,32)
        self.conv1_3 = convs(32,1,0,1)
        #第二层
        self.conv2_1 = convs(256,128,0,1)
        self.conv2_2 = deconvs(128,64)
        self.conv2_3 = convs(64,1,1,3)
        #第三层
        self.conv3_1 = convs(512,128,0,1)
        self.conv3_2 = deconvs(128,128)
        self.conv3_3 = deconvs(128,64)
        self.conv3_4 = convs(64,1,1,3)       
        
        #第四层
        self.conv4_1 = convs(512,128,0,1)
        self.conv4_2 = deconvs(128,128)
        self.conv4_3 = deconvs(128,64)
        self.conv4_4 = convs(64,1,1,3)
        
        #第五层
        self.conv5_1 = convs(1024,128,0,1)
        self.conv5_2 = deconvs(128,128)
        self.conv5_3 = deconvs(128,128)
        self.conv5_4 = deconvs(128,64)
        self.conv5_5 = convs(64,1,1,3)
        
        #第六层
        self.conv6_1 = convs(1024,128,0,1)
        self.conv6_2 = deconvs(128,128)
        self.conv6_3 = deconvs(128,128)
        self.conv6_4 = deconvs(128,64)
        self.conv6_5 = convs(64,1,1,3)
        
        
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
class ResnetMulFeatureWeightNetL(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNetL, self).__init__()
        model = timm.create_model('resnet50', pretrained=pretrained)
        extractor_blocks = [
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.act1,
                model.maxpool,
            ),
            model.layer1,
            model.layer2,
            model.layer3[0:3],
            model.layer3[3:],
            model.layer4[:3],
            ]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,32,0,1)
        self.conv1_2 = deconvs(32,32)
        self.conv1_3 = convs(32,1,0,1)
        #第二层
        self.conv2_1 = convs(256,128,0,1)
        self.conv2_2 = deconvs(128,64)
        self.conv2_3 = convs(64,1,1,3)
        #第三层
        self.conv3_1 = convs(512,128,0,1)
        self.conv3_2 = deconvs(128,128)
        self.conv3_3 = deconvs(128,64)
        self.conv3_4 = convs(64,1,1,3)       
        
        #第四层
        self.conv4_1 = convs(1024,128,0,1)
        self.conv4_2 = deconvs(128,128)
        self.conv4_3 = deconvs(128,128)
        self.conv4_4 = deconvs(128,64)
        self.conv4_5 = convs(64,1,1,3)
        
        #第五层
        self.conv5_1 = convs(1024,128,0,1)
        self.conv5_2 = deconvs(128,128)
        self.conv5_3 = deconvs(128,128)
        self.conv5_4 = deconvs(128,64)
        self.conv5_5 = convs(64,1,1,3)
        
        #第六层
        self.conv6_1 = convs(2048,128,0,1)
        self.conv6_2 = deconvs(128,128)
        self.conv6_3 = deconvs(128,128)
        self.conv6_4 = deconvs(128,128)
        self.conv6_5 = deconvs(128,64)
        self.conv6_6 = convs(64,1,1,3)
        
        
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
class ResnetMulFeatureWeightNetL_smallsize(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNetL_smallsize, self).__init__()
        model = timm.create_model('resnet50', pretrained=pretrained)
        extractor_blocks = [
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.act1,
                model.maxpool,
            ),
            model.layer1,
            model.layer2,
            model.layer3[0:3],
            model.layer3[3:],
            model.layer4[:3],
            ]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,32,0,1)
        self.conv1_2 = convs(32,32,1,3)
        self.conv1_3 = convs(32,1,0,1)
        #第二层
        self.conv2_1 = convs(256,64,0,1)
        self.conv2_2 = convs(64,64,1,3)
        self.conv2_3 = convs(64,1,0,1)
        #第三层
        self.conv3_1 = convs(512,128,0,1)
        self.conv3_3 = deconvs(128,64)
        self.conv3_4 = convs(64,1,1,3)       
        
        #第四层
        self.conv4_1 = convs(1024,128,0,1)
        self.conv4_3 = deconvs(128,128)
        self.conv4_4 = deconvs(128,64)
        self.conv4_5 = convs(64,1,1,3)
        
        #第五层
        self.conv5_1 = convs(1024,128,0,1)
        self.conv5_3 = deconvs(128,128)
        self.conv5_4 = deconvs(128,64)
        self.conv5_5 = convs(64,1,1,3)
        
        #第六层
        self.conv6_1 = convs(2048,128,0,1)
        self.conv6_3 = deconvs(128,128)
        self.conv6_4 = deconvs(128,128)
        self.conv6_5 = deconvs(128,64)
        self.conv6_6 = convs(64,1,1,3)
        
        
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
        concat_feature_3 = self.conv3_3(concat_feature_3)
        concat_feature_3 = self.conv3_4(concat_feature_3)
        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.conv4_1(concat_feature_4)
        concat_feature_4 = self.conv4_3(concat_feature_4)
        concat_feature_4 = self.conv4_4(concat_feature_4)
        concat_feature_4 = self.conv4_5(concat_feature_4)
        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.conv5_1(concat_feature_5)
        concat_feature_5 = self.conv5_3(concat_feature_5)
        concat_feature_5 = self.conv5_4(concat_feature_5)
        concat_feature_5 = self.conv5_5(concat_feature_5)
        
        
        
        concat_feature_6 = dis_feature[5]
        concat_feature_6 = self.conv6_1(concat_feature_6)
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
class ResnetMulFeatureWeightNetXL(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetMulFeatureWeightNetXL, self).__init__()
        model = timm.create_model('resnet50', pretrained=pretrained)
        extractor_blocks = [
           nn.Sequential(
                model.conv1,
                model.bn1,
                model.act1,
                model.maxpool,
            ),
            model.layer1,
            model.layer2[0:3],
            nn.Sequential(
                model.layer2[3:],
                model.layer3[0:2],
            ),
            model.layer3[2:5],
            nn.Sequential(
                model.layer3[5:],
                model.layer4,
            )
            ]
        self.Extractor = ResNetFeatureExtractor(extractor_blocks)
        #第一层
        self.conv1_1 = convs(64,32,0,1)
        self.conv1_2 = deconvs(32,32)
        self.conv1_3 = convs(32,1,0,1)
        #第二层
        self.conv2_1 = convs(256,128,0,1)
        self.conv2_2 = deconvs(128,64)
        self.conv2_3 = convs(64,1,1,3)
        #第三层
        self.conv3_1 = convs(512,128,0,1)
        self.conv3_2 = deconvs(128,128)
        self.conv3_3 = deconvs(128,64)
        self.conv3_4 = convs(64,1,1,3)       
        
        #第四层
        self.conv4_1 = convs(1024,128,0,1)
        self.conv4_2 = deconvs(128,128)
        self.conv4_3 = deconvs(128,128)
        self.conv4_4 = deconvs(128,64)
        self.conv4_5 = convs(64,1,1,3)
        
        #第五层
        self.conv5_1 = convs(1024,128,0,1)
        self.conv5_2 = deconvs(128,128)
        self.conv5_3 = deconvs(128,128)
        self.conv5_4 = deconvs(128,64)
        self.conv5_5 = convs(64,1,1,3)
        
        #第六层
        self.conv6_1 = convs(2048,128,0,1)
        self.conv6_2 = deconvs(128,128)
        self.conv6_3 = deconvs(128,128)
        self.conv6_4 = deconvs(128,128)
        self.conv6_5 = deconvs(128,64)
        self.conv6_6 = convs(64,1,1,3)
        
        
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
