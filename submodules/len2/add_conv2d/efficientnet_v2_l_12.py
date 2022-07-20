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
        self.conv2d27 = Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x95, x89):
        x96=operator.add(x95, x89)
        x97=self.conv2d27(x96)
        return x97

m = M().eval()
x95 = torch.randn(torch.Size([1, 96, 28, 28]))
x89 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x95, x89)
end = time.time()
print(end-start)
