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
        self.conv2d54 = Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x177):
        x190=self.conv2d54(x177)
        x191=self.batchnorm2d54(x190)
        x192=torch.nn.functional.relu(x191,inplace=True)
        return x192

m = M().eval()
x177 = torch.randn(torch.Size([1, 768, 12, 12]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
