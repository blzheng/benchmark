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
        self.batchnorm2d50 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)

    def forward(self, x252):
        x253=self.batchnorm2d50(x252)
        x254=self.relu61(x253)
        return x254

m = M().eval()
x252 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x252)
end = time.time()
print(end-start)
