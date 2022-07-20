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
        self.batchnorm2d50 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x183):
        x184=self.batchnorm2d50(x183)
        x185=torch.nn.functional.relu(x184,inplace=True)
        return x185

m = M().eval()
x183 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
