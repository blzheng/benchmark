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
        self.conv2d106 = Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24), bias=False)
        self.batchnorm2d106 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x344):
        x351=self.conv2d106(x344)
        x352=self.batchnorm2d106(x351)
        return x352

m = M().eval()
x344 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
