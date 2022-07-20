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
        self.conv2d51 = Conv2d(896, 224, kernel_size=(1, 1), stride=(1, 1))
        self.relu39 = ReLU()
        self.conv2d52 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x160):
        x161=self.conv2d51(x160)
        x162=self.relu39(x161)
        x163=self.conv2d52(x162)
        x164=self.sigmoid9(x163)
        return x164

m = M().eval()
x160 = torch.randn(torch.Size([1, 896, 1, 1]))
start = time.time()
output = m(x160)
end = time.time()
print(end-start)