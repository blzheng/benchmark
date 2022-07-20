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
        self.batchnorm2d178 = BatchNorm2d(1568, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu178 = ReLU(inplace=True)

    def forward(self, x629):
        x630=self.batchnorm2d178(x629)
        x631=self.relu178(x630)
        return x631

m = M().eval()
x629 = torch.randn(torch.Size([1, 1568, 7, 7]))
start = time.time()
output = m(x629)
end = time.time()
print(end-start)
