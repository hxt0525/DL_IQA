from .cnniqa import CNNIQA
from .multifeatureiqa import MultiFeature_IQA,MultiFeature_IQA2
from .EfficientMulFeatureNet import *
from .ResNetMulFeatureNet import *
from .decoder_unet import *
__all__ = [
    'CNNIQA','MultiFeature_IQA','MultiFeature_IQA2','EfficientB2MulFeatureWeightNetMall','EfficientB2MulFeatureWeightNet','EfficientB2MulFeatureWeightNetMallX',
    'EfficientB2MulFeatureNetMall','EfficientB2MulFeatureNet','ResnetMulFeatureWeightNet','EfficientB3MulFeatureWeightNetMall',
    'EfficientB2MulFeatureWeightNetMallCBAM','EffB2_UNetPlusPlusPrune','EffB2_UNetPrune','EffB2_FPNPrune','EfficientB1MulFeatureWeightNetMall','EffB2_UNetRestore',
    'EfficientB2MulFeatureWeightNetMall_addstd','ResnetMulFeatureWeightNetS'
]