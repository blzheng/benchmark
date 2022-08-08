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
        self.batchnorm2d90 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)

    def forward(self, x299, x292):
        x300=self.batchnorm2d90(x299)
        x301=operator.add(x300, x292)
        x302=self.relu85(x301)
        return x302

m = M().eval()
x299 = torch.randn(torch.Size([1, 1024, 28, 28]))
x292 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x299, x292)
end = time.time()
print(end-start)
