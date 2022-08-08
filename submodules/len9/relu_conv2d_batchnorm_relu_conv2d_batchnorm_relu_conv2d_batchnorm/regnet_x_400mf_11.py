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
        self.relu45 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d51 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu47 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x160):
        x161=self.relu45(x160)
        x162=self.conv2d50(x161)
        x163=self.batchnorm2d50(x162)
        x164=self.relu46(x163)
        x165=self.conv2d51(x164)
        x166=self.batchnorm2d51(x165)
        x167=self.relu47(x166)
        x168=self.conv2d52(x167)
        x169=self.batchnorm2d52(x168)
        return x169

m = M().eval()
x160 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x160)
end = time.time()
print(end-start)
