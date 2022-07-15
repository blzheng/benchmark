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
        self.conv2d58 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)

    def forward(self, x175):
        x176=self.conv2d58(x175)
        return x176

m = M().eval()
x175 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
