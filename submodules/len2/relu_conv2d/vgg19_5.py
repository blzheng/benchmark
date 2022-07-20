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
        self.relu8 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x20):
        x21=self.relu8(x20)
        x22=self.conv2d9(x21)
        return x22

m = M().eval()
x20 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x20)
end = time.time()
print(end-start)
