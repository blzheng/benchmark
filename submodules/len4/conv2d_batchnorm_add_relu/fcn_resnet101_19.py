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
        self.conv2d54 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)

    def forward(self, x178, x172):
        x179=self.conv2d54(x178)
        x180=self.batchnorm2d54(x179)
        x181=operator.add(x180, x172)
        x182=self.relu49(x181)
        return x182

m = M().eval()
x178 = torch.randn(torch.Size([1, 256, 28, 28]))
x172 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x178, x172)
end = time.time()
print(end-start)
