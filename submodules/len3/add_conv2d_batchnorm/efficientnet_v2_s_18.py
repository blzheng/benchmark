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
        self.conv2d84 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x268, x253):
        x269=operator.add(x268, x253)
        x270=self.conv2d84(x269)
        x271=self.batchnorm2d58(x270)
        return x271

m = M().eval()
x268 = torch.randn(torch.Size([1, 160, 14, 14]))
x253 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x268, x253)
end = time.time()
print(end-start)
