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
        self.relu70 = ReLU(inplace=True)
        self.conv2d76 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)
        self.conv2d77 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d77 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x249):
        x250=self.relu70(x249)
        x251=self.conv2d76(x250)
        x252=self.batchnorm2d76(x251)
        x253=self.relu73(x252)
        x254=self.conv2d77(x253)
        x255=self.batchnorm2d77(x254)
        return x255

m = M().eval()
x249 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x249)
end = time.time()
print(end-start)
