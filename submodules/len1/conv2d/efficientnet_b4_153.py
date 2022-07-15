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
        self.conv2d153 = Conv2d(1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x477):
        x478=self.conv2d153(x477)
        return x478

m = M().eval()
x477 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x477)
end = time.time()
print(end-start)
