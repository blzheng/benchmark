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
        self.batchnorm2d9 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x45):
        x46=self.batchnorm2d9(x45)
        x47=torch.nn.functional.relu(x46,inplace=True)
        return x47

m = M().eval()
x45 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
