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
        self.batchnorm2d92 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x315):
        x316=self.batchnorm2d92(x315)
        x317=torch.nn.functional.relu(x316,inplace=True)
        return x317

m = M().eval()
x315 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x315)
end = time.time()
print(end-start)
