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
        self.relu632 = ReLU6(inplace=True)
        self.conv2d49 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)

    def forward(self, x140):
        x141=self.relu632(x140)
        x142=self.conv2d49(x141)
        return x142

m = M().eval()
x140 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x140)
end = time.time()
print(end-start)