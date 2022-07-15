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
        self.batchnorm2d149 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x752):
        x753=self.batchnorm2d149(x752)
        return x753

m = M().eval()
x752 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x752)
end = time.time()
print(end-start)
