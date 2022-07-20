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
        self.conv2d14 = Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)

    def forward(self, x58):
        x59=torch.nn.functional.relu(x58,inplace=True)
        x60=self.conv2d14(x59)
        return x60

m = M().eval()
x58 = torch.randn(torch.Size([1, 48, 25, 25]))
start = time.time()
output = m(x58)
end = time.time()
print(end-start)
