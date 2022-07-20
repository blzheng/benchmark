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
        self.batchnorm2d146 = BatchNorm2d(1872, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu146 = ReLU(inplace=True)

    def forward(self, x517):
        x518=self.batchnorm2d146(x517)
        x519=self.relu146(x518)
        return x519

m = M().eval()
x517 = torch.randn(torch.Size([1, 1872, 7, 7]))
start = time.time()
output = m(x517)
end = time.time()
print(end-start)
