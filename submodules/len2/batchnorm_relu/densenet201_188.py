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
        self.batchnorm2d188 = BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu188 = ReLU(inplace=True)

    def forward(self, x664):
        x665=self.batchnorm2d188(x664)
        x666=self.relu188(x665)
        return x666

m = M().eval()
x664 = torch.randn(torch.Size([1, 1728, 7, 7]))
start = time.time()
output = m(x664)
end = time.time()
print(end-start)
