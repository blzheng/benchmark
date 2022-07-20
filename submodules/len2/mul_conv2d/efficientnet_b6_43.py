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
        self.conv2d217 = Conv2d(3456, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x679, x674):
        x680=operator.mul(x679, x674)
        x681=self.conv2d217(x680)
        return x681

m = M().eval()
x679 = torch.randn(torch.Size([1, 3456, 1, 1]))
x674 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x679, x674)
end = time.time()
print(end-start)
