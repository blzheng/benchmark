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
        self.conv2d109 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d109 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu105 = ReLU()
        self.dropout0 = Dropout(p=0.5, inplace=False)

    def forward(self, x364):
        x365=self.conv2d109(x364)
        x366=self.batchnorm2d109(x365)
        x367=self.relu105(x366)
        x368=self.dropout0(x367)
        return x368

m = M().eval()
x364 = torch.randn(torch.Size([1, 1280, 28, 28]))
start = time.time()
output = m(x364)
end = time.time()
print(end-start)
