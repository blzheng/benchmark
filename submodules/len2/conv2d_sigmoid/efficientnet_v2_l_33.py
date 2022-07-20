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
        self.conv2d201 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid33 = Sigmoid()

    def forward(self, x648):
        x649=self.conv2d201(x648)
        x650=self.sigmoid33(x649)
        return x650

m = M().eval()
x648 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x648)
end = time.time()
print(end-start)
