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
        self.conv2d193 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d115 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x575):
        x576=self.conv2d193(x575)
        x577=self.batchnorm2d115(x576)
        return x577

m = M().eval()
x575 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x575)
end = time.time()
print(end-start)
