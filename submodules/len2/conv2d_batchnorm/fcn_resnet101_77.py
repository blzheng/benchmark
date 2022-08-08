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
        self.conv2d77 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d77 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x255):
        x256=self.conv2d77(x255)
        x257=self.batchnorm2d77(x256)
        return x257

m = M().eval()
x255 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x255)
end = time.time()
print(end-start)
