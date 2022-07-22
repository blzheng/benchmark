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
        self.conv2d49 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d50 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x160):
        x161=self.conv2d49(x160)
        x162=self.batchnorm2d49(x161)
        x163=self.relu46(x162)
        x164=self.conv2d50(x163)
        x165=self.batchnorm2d50(x164)
        x166=self.relu46(x165)
        return x166

m = M().eval()
x160 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x160)
end = time.time()
print(end-start)
