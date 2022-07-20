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
        self.conv2d33 = Conv2d(384, 384, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=384)

    def forward(self, x347, x337):
        x348=operator.add(x347, x337)
        x350=self.conv2d33(x348)
        return x350

m = M().eval()
x347 = torch.randn(torch.Size([1, 384, 14, 14]))
x337 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x347, x337)
end = time.time()
print(end-start)
