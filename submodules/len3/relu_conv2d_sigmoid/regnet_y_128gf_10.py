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
        self.relu43 = ReLU()
        self.conv2d57 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()

    def forward(self, x177):
        x178=self.relu43(x177)
        x179=self.conv2d57(x178)
        x180=self.sigmoid10(x179)
        return x180

m = M().eval()
x177 = torch.randn(torch.Size([1, 726, 1, 1]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
