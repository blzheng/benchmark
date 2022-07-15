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
        self.batchnorm2d155 = BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x781):
        x782=self.batchnorm2d155(x781)
        return x782

m = M().eval()
x781 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x781)
end = time.time()
print(end-start)
