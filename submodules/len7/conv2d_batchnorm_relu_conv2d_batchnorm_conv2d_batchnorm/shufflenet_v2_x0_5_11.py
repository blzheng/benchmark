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
        self.conv2d38 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d39 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d40 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x253):
        x254=self.conv2d38(x253)
        x255=self.batchnorm2d38(x254)
        x256=self.relu25(x255)
        x257=self.conv2d39(x256)
        x258=self.batchnorm2d39(x257)
        x259=self.conv2d40(x258)
        x260=self.batchnorm2d40(x259)
        return x260

m = M().eval()
x253 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)