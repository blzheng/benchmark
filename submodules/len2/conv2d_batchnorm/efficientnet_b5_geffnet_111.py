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
        self.conv2d187 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x557):
        x558=self.conv2d187(x557)
        x559=self.batchnorm2d111(x558)
        return x559

m = M().eval()
x557 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x557)
end = time.time()
print(end-start)
