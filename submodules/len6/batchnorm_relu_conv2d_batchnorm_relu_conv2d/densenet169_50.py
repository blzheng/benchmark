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
        self.batchnorm2d104 = BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu104 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu105 = ReLU(inplace=True)
        self.conv2d105 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x370):
        x371=self.batchnorm2d104(x370)
        x372=self.relu104(x371)
        x373=self.conv2d104(x372)
        x374=self.batchnorm2d105(x373)
        x375=self.relu105(x374)
        x376=self.conv2d105(x375)
        return x376

m = M().eval()
x370 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x370)
end = time.time()
print(end-start)
