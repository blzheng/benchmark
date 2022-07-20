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
        self.conv2d191 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid38 = Sigmoid()

    def forward(self, x597, x594):
        x598=self.conv2d191(x597)
        x599=self.sigmoid38(x598)
        x600=operator.mul(x599, x594)
        return x600

m = M().eval()
x597 = torch.randn(torch.Size([1, 128, 1, 1]))
x594 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x597, x594)
end = time.time()
print(end-start)
