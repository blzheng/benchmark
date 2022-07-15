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
        self.conv2d30 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)

    def forward(self, x92):
        x93=self.conv2d30(x92)
        return x93

m = M().eval()
x92 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
