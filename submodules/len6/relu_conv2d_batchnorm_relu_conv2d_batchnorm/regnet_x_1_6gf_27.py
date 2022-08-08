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
        self.relu45 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=17, bias=False)
        self.batchnorm2d50 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x158):
        x159=self.relu45(x158)
        x160=self.conv2d49(x159)
        x161=self.batchnorm2d49(x160)
        x162=self.relu46(x161)
        x163=self.conv2d50(x162)
        x164=self.batchnorm2d50(x163)
        return x164

m = M().eval()
x158 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x158)
end = time.time()
print(end-start)
