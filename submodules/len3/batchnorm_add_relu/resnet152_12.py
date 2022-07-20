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
        self.batchnorm2d35 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)

    def forward(self, x115, x108):
        x116=self.batchnorm2d35(x115)
        x117=operator.add(x116, x108)
        x118=self.relu31(x117)
        return x118

m = M().eval()
x115 = torch.randn(torch.Size([1, 512, 28, 28]))
x108 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x115, x108)
end = time.time()
print(end-start)
