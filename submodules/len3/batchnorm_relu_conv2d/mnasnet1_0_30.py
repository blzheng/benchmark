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
        self.batchnorm2d45 = BatchNorm2d(1152, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)

    def forward(self, x130):
        x131=self.batchnorm2d45(x130)
        x132=self.relu30(x131)
        x133=self.conv2d46(x132)
        return x133

m = M().eval()
x130 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x130)
end = time.time()
print(end-start)
