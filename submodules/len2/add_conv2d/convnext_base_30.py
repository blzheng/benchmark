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
        self.conv2d38 = Conv2d(1024, 1024, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1024)

    def forward(self, x397, x387):
        x398=operator.add(x397, x387)
        x400=self.conv2d38(x398)
        return x400

m = M().eval()
x397 = torch.randn(torch.Size([1, 1024, 7, 7]))
x387 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x397, x387)
end = time.time()
print(end-start)
