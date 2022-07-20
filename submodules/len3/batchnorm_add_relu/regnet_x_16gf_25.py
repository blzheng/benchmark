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
        self.batchnorm2d70 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU(inplace=True)

    def forward(self, x228, x221):
        x229=self.batchnorm2d70(x228)
        x230=operator.add(x221, x229)
        x231=self.relu66(x230)
        return x231

m = M().eval()
x228 = torch.randn(torch.Size([1, 2048, 7, 7]))
x221 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x228, x221)
end = time.time()
print(end-start)
