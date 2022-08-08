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
        self.relu79 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x277):
        x278=self.relu79(x277)
        x279=self.conv2d84(x278)
        x280=self.batchnorm2d84(x279)
        return x280

m = M().eval()
x277 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x277)
end = time.time()
print(end-start)
