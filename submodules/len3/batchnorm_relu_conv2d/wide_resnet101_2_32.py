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
        self.batchnorm2d52 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x171):
        x172=self.batchnorm2d52(x171)
        x173=self.relu49(x172)
        x174=self.conv2d53(x173)
        return x174

m = M().eval()
x171 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x171)
end = time.time()
print(end-start)
