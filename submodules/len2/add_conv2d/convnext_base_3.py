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
        self.conv2d7 = Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=256)

    def forward(self, x66, x56):
        x67=operator.add(x66, x56)
        x69=self.conv2d7(x67)
        return x69

m = M().eval()
x66 = torch.randn(torch.Size([1, 256, 28, 28]))
x56 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x66, x56)
end = time.time()
print(end-start)
