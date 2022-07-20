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
        self.batchnorm2d131 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu131 = ReLU(inplace=True)

    def forward(self, x464):
        x465=self.batchnorm2d131(x464)
        x466=self.relu131(x465)
        return x466

m = M().eval()
x464 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x464)
end = time.time()
print(end-start)
