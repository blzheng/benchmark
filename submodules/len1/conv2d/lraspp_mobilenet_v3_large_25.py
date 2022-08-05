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
        self.conv2d25 = Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)

    def forward(self, x73):
        x74=self.conv2d25(x73)
        return x74

m = M().eval()
x73 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x73)
end = time.time()
print(end-start)
