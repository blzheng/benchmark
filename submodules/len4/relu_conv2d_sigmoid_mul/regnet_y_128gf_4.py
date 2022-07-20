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
        self.relu19 = ReLU()
        self.conv2d26 = Conv2d(264, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()

    def forward(self, x79, x77):
        x80=self.relu19(x79)
        x81=self.conv2d26(x80)
        x82=self.sigmoid4(x81)
        x83=operator.mul(x82, x77)
        return x83

m = M().eval()
x79 = torch.randn(torch.Size([1, 264, 1, 1]))
x77 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x79, x77)
end = time.time()
print(end-start)
