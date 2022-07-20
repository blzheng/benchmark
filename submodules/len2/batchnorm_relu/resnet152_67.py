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
        self.batchnorm2d103 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu100 = ReLU(inplace=True)

    def forward(self, x341):
        x342=self.batchnorm2d103(x341)
        x343=self.relu100(x342)
        return x343

m = M().eval()
x341 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x341)
end = time.time()
print(end-start)
