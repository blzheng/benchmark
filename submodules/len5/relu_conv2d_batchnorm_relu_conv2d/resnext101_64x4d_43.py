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
        self.relu64 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)

    def forward(self, x229):
        x230=self.relu64(x229)
        x231=self.conv2d70(x230)
        x232=self.batchnorm2d70(x231)
        x233=self.relu67(x232)
        x234=self.conv2d71(x233)
        return x234

m = M().eval()
x229 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x229)
end = time.time()
print(end-start)
