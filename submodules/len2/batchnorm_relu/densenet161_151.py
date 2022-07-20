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
        self.batchnorm2d151 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu151 = ReLU(inplace=True)

    def forward(self, x534):
        x535=self.batchnorm2d151(x534)
        x536=self.relu151(x535)
        return x536

m = M().eval()
x534 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x534)
end = time.time()
print(end-start)