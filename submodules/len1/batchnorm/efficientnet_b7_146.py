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
        self.batchnorm2d146 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x776):
        x777=self.batchnorm2d146(x776)
        return x777

m = M().eval()
x776 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x776)
end = time.time()
print(end-start)
