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
        self.batchnorm2d67 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)

    def forward(self, x220):
        x221=self.batchnorm2d67(x220)
        x222=self.relu64(x221)
        return x222

m = M().eval()
x220 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)