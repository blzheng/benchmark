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
        self.batchnorm2d92 = BatchNorm2d(2688, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x458):
        x459=self.batchnorm2d92(x458)
        return x459

m = M().eval()
x458 = torch.randn(torch.Size([1, 2688, 7, 7]))
start = time.time()
output = m(x458)
end = time.time()
print(end-start)
