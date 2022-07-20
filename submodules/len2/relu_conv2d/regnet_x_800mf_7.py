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
        self.relu7 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)

    def forward(self, x29):
        x30=self.relu7(x29)
        x31=self.conv2d10(x30)
        return x31

m = M().eval()
x29 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
