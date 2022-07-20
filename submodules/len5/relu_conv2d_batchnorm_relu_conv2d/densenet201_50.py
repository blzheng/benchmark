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
        self.relu103 = ReLU(inplace=True)
        self.conv2d103 = Conv2d(1280, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d104 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu104 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x366):
        x367=self.relu103(x366)
        x368=self.conv2d103(x367)
        x369=self.batchnorm2d104(x368)
        x370=self.relu104(x369)
        x371=self.conv2d104(x370)
        return x371

m = M().eval()
x366 = torch.randn(torch.Size([1, 1280, 14, 14]))
start = time.time()
output = m(x366)
end = time.time()
print(end-start)
