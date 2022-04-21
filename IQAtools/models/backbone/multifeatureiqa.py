# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:46:13 2021

@author: HXT
"""

import torch
import torch.nn as nn
import torchvision
from ..builder import Model_List
import torch.nn.functional as F
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


@Model_List.insert_Module()
class MultiFeature_IQA(nn.Module):
    def __init__(self):
        super(MultiFeature_IQA, self).__init__()
        self.resnet = torchvision.models.resnet.resnet50(pretrained=True)
        self.feature_extractor_blocks = [
            # conv1
            nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu
            ),
            # conv2_x
            nn.Sequential(
                self.resnet.maxpool,
                self.resnet.layer1
            ),
            # conv3_x
            self.resnet.layer2,
            # conv4_x
            self.resnet.layer3,
            # conv5_x
            self.resnet.layer4,]
        self.Extractor = ResNetFeatureExtractor(self.feature_extractor_blocks)
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
        
        
        #第五层
        self.conv5_1 = convs(2048,1024,0,1)
        self.conv5_2 = deconvs(1024,512,2,2)
        self.conv5_3 = deconvs(512,256,2,2)
        self.conv5_4 = deconvs(256,128,2,2)
        self.conv5_5 = deconvs(128,128,2,2)
        self.conv5_6 = convs(128,1,1,3)
        
        #特征汇聚层
        self.conv6_1 = convs(5,32,0,1)
        self.conv6_2 = convs(32,16,1,3)
        self.conv6_3 = convs(16,1,1,3)
        
        #分数
        self.AdPool = nn.AdaptiveAvgPool2d((16,16))
        self.Linear = nn.Linear(256, 1)
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
        concat_feature_5 = self.conv5_6(concat_feature_5)
        #print(concat_feature_5.shape)
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,
                                          concat_feature_3,concat_feature_4,
                                          concat_feature_5],1)
        
        #print(concat_feature_again.shape)
        concat_feature_again = self.conv6_1(concat_feature_again)
        concat_feature_again = self.conv6_2(concat_feature_again)
        concat_feature_again = self.conv6_3(concat_feature_again)
        
        final_scores = self.AdPool(concat_feature_again)
        final_scores = final_scores.view(final_scores.shape[0],-1)
        final_scores = self.Linear(final_scores)
        return final_scores



@Model_List.insert_Module()
class MultiFeature_IQA2(nn.Module):
    def __init__(self):
        super(MultiFeature_IQA2, self).__init__()
        self.resnet = torchvision.models.resnet.resnet50(pretrained=True)
        self.feature_extractor_blocks = [
            # conv1
            nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu
            ),
            # conv2_x
            nn.Sequential(
                self.resnet.maxpool,
                self.resnet.layer1
            ),
            # conv3_x
            self.resnet.layer2,
            # conv4_x
            self.resnet.layer3,
            # conv5_x
            self.resnet.layer4,]
        self.Extractor = ResNetFeatureExtractor(self.feature_extractor_blocks)
        #第一层
        self.AdpPool_1 = nn.AdaptiveAvgPool2d((8,8))
        #第二层
        self.AdpPool_2 = nn.AdaptiveAvgPool2d((8,8))
        #第三层
        self.AdpPool_3 = nn.AdaptiveAvgPool2d((8,8))
        #第四层
        self.AdpPool_4 = nn.AdaptiveAvgPool2d((8,8))
        #第五层
        self.AdpPool_5 = nn.AdaptiveAvgPool2d((8,8))
        
        #特征汇聚
        self.FeatureSum = convs(3904,1024,0,1)
        self.MaxPool = nn.AdaptiveMaxPool2d((1,1))
        self.AvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.Linear1 = nn.Linear(2048, 1024)
        
        self.Linear2 = nn.Linear(1024, 1)
    def forward(self,dis):
        dis_feature = self.Extractor(dis)
        
        concat_feature_1 = dis_feature[0]
        concat_feature_1 = self.AdpPool_1(concat_feature_1)
        #print(concat_feature_1.shape)
        
        concat_feature_2 = dis_feature[1]
        concat_feature_2 = self.AdpPool_2(concat_feature_2)

        #print(concat_feature_2.shape)
        
        concat_feature_3 = dis_feature[2]
        concat_feature_3 = self.AdpPool_3(concat_feature_3)

        #print(concat_feature_3.shape)
        
        concat_feature_4 = dis_feature[3]
        concat_feature_4 = self.AdpPool_4(concat_feature_4)

        #print(concat_feature_4.shape)
        
        concat_feature_5 = dis_feature[4]
        concat_feature_5 = self.AdpPool_5(concat_feature_5)
        
        #print(concat_feature_5.shape)
        
        concat_feature_again = torch.cat([concat_feature_1,concat_feature_2,
                                          concat_feature_3,concat_feature_4,
                                          concat_feature_5],1)
        
        feature_sum = self.FeatureSum(concat_feature_again)
        
        f1 = self.MaxPool(feature_sum)
        f2 = self.AvgPool(feature_sum)
        
        f = torch.cat([f1,f2],1)
        f = f.view(f.shape[0],-1)
        
        final_scores = F.relu(self.Linear1(f))
        final_scores = F.dropout(final_scores)
        final_scores = self.Linear2(final_scores)


        return final_scores

'''
model = MultiFeature_IQA()
inputs = torch.zeros((4,3,128,128))
outs = model(inputs)
print(outs.shape)
'''