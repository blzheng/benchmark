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
        self.conv2d184 = Conv2d(1344, 56, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x551):
        x552=x551.mean((2, 3),keepdim=True)
        x553=self.conv2d184(x552)
        return x553

m = M().eval()
x551 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x551)
end = time.time()
print(end-start)