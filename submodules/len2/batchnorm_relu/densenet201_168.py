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
        self.batchnorm2d168 = BatchNorm2d(1408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu168 = ReLU(inplace=True)

    def forward(self, x594):
        x595=self.batchnorm2d168(x594)
        x596=self.relu168(x595)
        return x596

m = M().eval()
x594 = torch.randn(torch.Size([1, 1408, 7, 7]))
start = time.time()
output = m(x594)
end = time.time()
print(end-start)
