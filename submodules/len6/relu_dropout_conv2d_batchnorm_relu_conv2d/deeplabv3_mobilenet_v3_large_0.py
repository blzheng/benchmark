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
        self.relu24 = ReLU()
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.conv2d68 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU()
        self.conv2d69 = Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x205):
        x206=self.relu24(x205)
        x207=self.dropout0(x206)
        x208=self.conv2d68(x207)
        x209=self.batchnorm2d52(x208)
        x210=self.relu25(x209)
        x211=self.conv2d69(x210)
        return x211

m = M().eval()
x205 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x205)
end = time.time()
print(end-start)
