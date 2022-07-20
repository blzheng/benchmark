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
        self.batchnorm2d62 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu78 = ReLU(inplace=True)

    def forward(self, x317):
        x318=self.batchnorm2d62(x317)
        x319=self.relu78(x318)
        return x319

m = M().eval()
x317 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x317)
end = time.time()
print(end-start)
