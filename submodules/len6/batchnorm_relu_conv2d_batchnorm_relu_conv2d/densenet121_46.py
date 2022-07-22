import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.batchnorm2d96 = BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu96 = ReLU(inplace=True)
        self.conv2d96 = Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d97 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)
        self.conv2d97 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x342):
        x343=self.batchnorm2d96(x342)
        x344=self.relu96(x343)
        x345=self.conv2d96(x344)
        x346=self.batchnorm2d97(x345)
        x347=self.relu97(x346)
        x348=self.conv2d97(x347)
        return x348

m = M().eval()
x342 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x342)
end = time.time()
print(end-start)
