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
        self.batchnorm2d48 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x176):
        x177=self.batchnorm2d48(x176)
        x178=torch.nn.functional.relu(x177,inplace=True)
        x179=self.conv2d49(x178)
        return x179

m = M().eval()
x176 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
