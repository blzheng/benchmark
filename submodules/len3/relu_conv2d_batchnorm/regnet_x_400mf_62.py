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
        self.relu62 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x216):
        x217=self.relu62(x216)
        x218=self.conv2d67(x217)
        x219=self.batchnorm2d67(x218)
        return x219

m = M().eval()
x216 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x216)
end = time.time()
print(end-start)
