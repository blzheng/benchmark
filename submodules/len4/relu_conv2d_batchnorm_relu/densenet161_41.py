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
        self.relu85 = ReLU(inplace=True)
        self.conv2d85 = Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d86 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu86 = ReLU(inplace=True)

    def forward(self, x303):
        x304=self.relu85(x303)
        x305=self.conv2d85(x304)
        x306=self.batchnorm2d86(x305)
        x307=self.relu86(x306)
        return x307

m = M().eval()
x303 = torch.randn(torch.Size([1, 1488, 14, 14]))
start = time.time()
output = m(x303)
end = time.time()
print(end-start)
