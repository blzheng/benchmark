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
        self.batchnorm2d52 = BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)

    def forward(self, x266, x281):
        x267=self.batchnorm2d52(x266)
        x282=operator.add(x267, x281)
        x283=self.relu68(x282)
        return x283

m = M().eval()
x266 = torch.randn(torch.Size([1, 2016, 7, 7]))
x281 = torch.randn(torch.Size([1, 2016, 7, 7]))
start = time.time()
output = m(x266, x281)
end = time.time()
print(end-start)
