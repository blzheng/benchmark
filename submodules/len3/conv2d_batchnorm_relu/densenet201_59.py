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
        self.conv2d119 = Conv2d(1536, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d120 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu120 = ReLU(inplace=True)

    def forward(self, x423):
        x424=self.conv2d119(x423)
        x425=self.batchnorm2d120(x424)
        x426=self.relu120(x425)
        return x426

m = M().eval()
x423 = torch.randn(torch.Size([1, 1536, 14, 14]))
start = time.time()
output = m(x423)
end = time.time()
print(end-start)
