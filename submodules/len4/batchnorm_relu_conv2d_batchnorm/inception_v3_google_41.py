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
        self.batchnorm2d74 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d75 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.batchnorm2d75 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x254):
        x255=self.batchnorm2d74(x254)
        x256=torch.nn.functional.relu(x255,inplace=True)
        x257=self.conv2d75(x256)
        x258=self.batchnorm2d75(x257)
        return x258

m = M().eval()
x254 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x254)
end = time.time()
print(end-start)
