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
        self.batchnorm2d9 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)

    def forward(self, x31, x30):
        x32=self.batchnorm2d9(x31)
        x33=operator.add(x30, x32)
        x34=self.relu7(x33)
        return x34

m = M().eval()
x31 = torch.randn(torch.Size([1, 128, 28, 28]))
x30 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x31, x30)
end = time.time()
print(end-start)
