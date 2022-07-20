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
        self.batchnorm2d67 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu63 = ReLU(inplace=True)

    def forward(self, x218, x211):
        x219=self.batchnorm2d67(x218)
        x220=operator.add(x211, x219)
        x221=self.relu63(x220)
        return x221

m = M().eval()
x218 = torch.randn(torch.Size([1, 400, 7, 7]))
x211 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x218, x211)
end = time.time()
print(end-start)
