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
        self.relu49 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d53 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x174):
        x175=self.relu49(x174)
        x176=self.conv2d53(x175)
        x177=self.batchnorm2d53(x176)
        x178=self.relu49(x177)
        x179=self.conv2d54(x178)
        return x179

m = M().eval()
x174 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
