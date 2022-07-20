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
        self.relu14 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)

    def forward(self, x62):
        x63=self.relu14(x62)
        x64=self.conv2d22(x63)
        return x64

m = M().eval()
x62 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x62)
end = time.time()
print(end-start)
