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
        self.conv2d30 = Conv2d(120, 30, kernel_size=(1, 1), stride=(1, 1))
        self.relu23 = ReLU()
        self.conv2d31 = Conv2d(30, 120, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()

    def forward(self, x94):
        x95=self.conv2d30(x94)
        x96=self.relu23(x95)
        x97=self.conv2d31(x96)
        x98=self.sigmoid5(x97)
        return x98

m = M().eval()
x94 = torch.randn(torch.Size([1, 120, 1, 1]))
start = time.time()
output = m(x94)
end = time.time()
print(end-start)
