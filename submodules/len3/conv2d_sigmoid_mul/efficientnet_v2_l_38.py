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
        self.conv2d226 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid38 = Sigmoid()

    def forward(self, x728, x725):
        x729=self.conv2d226(x728)
        x730=self.sigmoid38(x729)
        x731=operator.mul(x730, x725)
        return x731

m = M().eval()
x728 = torch.randn(torch.Size([1, 96, 1, 1]))
x725 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x728, x725)
end = time.time()
print(end-start)
