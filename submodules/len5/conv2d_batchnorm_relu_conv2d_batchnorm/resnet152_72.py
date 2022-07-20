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
        self.conv2d112 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d112 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu109 = ReLU(inplace=True)
        self.conv2d113 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d113 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x370):
        x371=self.conv2d112(x370)
        x372=self.batchnorm2d112(x371)
        x373=self.relu109(x372)
        x374=self.conv2d113(x373)
        x375=self.batchnorm2d113(x374)
        return x375

m = M().eval()
x370 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x370)
end = time.time()
print(end-start)
