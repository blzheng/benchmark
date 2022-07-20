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
        self.conv2d42 = Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d42 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x150):
        x151=torch.nn.functional.relu(x150,inplace=True)
        x152=self.conv2d42(x151)
        x153=self.batchnorm2d42(x152)
        return x153

m = M().eval()
x150 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x150)
end = time.time()
print(end-start)
