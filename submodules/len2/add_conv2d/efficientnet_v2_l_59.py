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
        self.conv2d268 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x862, x847):
        x863=operator.add(x862, x847)
        x864=self.conv2d268(x863)
        return x864

m = M().eval()
x862 = torch.randn(torch.Size([1, 384, 7, 7]))
x847 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x862, x847)
end = time.time()
print(end-start)
