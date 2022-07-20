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
        self.batchnorm2d58 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)

    def forward(self, x298):
        x299=self.batchnorm2d58(x298)
        x300=self.relu73(x299)
        return x300

m = M().eval()
x298 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x298)
end = time.time()
print(end-start)
