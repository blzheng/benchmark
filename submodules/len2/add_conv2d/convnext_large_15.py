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
        self.conv2d21 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x215, x205):
        x216=operator.add(x215, x205)
        x218=self.conv2d21(x216)
        return x218

m = M().eval()
x215 = torch.randn(torch.Size([1, 768, 14, 14]))
x205 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x215, x205)
end = time.time()
print(end-start)
