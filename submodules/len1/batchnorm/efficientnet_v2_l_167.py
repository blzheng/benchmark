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
        self.batchnorm2d167 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x828):
        x829=self.batchnorm2d167(x828)
        return x829

m = M().eval()
x828 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x828)
end = time.time()
print(end-start)
