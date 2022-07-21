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
        self.relu115 = ReLU(inplace=True)
        self.conv2d121 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d121 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu118 = ReLU(inplace=True)
        self.conv2d122 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d122 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d123 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x399):
        x400=self.relu115(x399)
        x401=self.conv2d121(x400)
        x402=self.batchnorm2d121(x401)
        x403=self.relu118(x402)
        x404=self.conv2d122(x403)
        x405=self.batchnorm2d122(x404)
        x406=self.relu118(x405)
        x407=self.conv2d123(x406)
        return x407

m = M().eval()
x399 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x399)
end = time.time()
print(end-start)
