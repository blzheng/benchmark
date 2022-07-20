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
        self.conv2d190 = Conv2d(1760, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d191 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x673):
        x674=self.conv2d190(x673)
        x675=self.batchnorm2d191(x674)
        return x675

m = M().eval()
x673 = torch.randn(torch.Size([1, 1760, 7, 7]))
start = time.time()
output = m(x673)
end = time.time()
print(end-start)
