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
import sys
import os

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        self.layernorm0 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.layernorm1 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.layernorm2 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.linear0 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu0 = GELU(approximate='none')
        self.dropout0 = Dropout(p=0.0, inplace=False)
        self.linear1 = Linear(in_features=384, out_features=96, bias=True)
        self.dropout1 = Dropout(p=0.0, inplace=False)
        self.layernorm3 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.layernorm4 = LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        self.linear2 = Linear(in_features=96, out_features=384, bias=True)
        self.gelu1 = GELU(approximate='none')
        self.dropout2 = Dropout(p=0.0, inplace=False)
        self.linear3 = Linear(in_features=384, out_features=96, bias=True)
        self.dropout3 = Dropout(p=0.0, inplace=False)
        self.layernorm5 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear4 = Linear(in_features=384, out_features=192, bias=False)
        self.layernorm6 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.layernorm7 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.linear5 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu2 = GELU(approximate='none')
        self.dropout4 = Dropout(p=0.0, inplace=False)
        self.linear6 = Linear(in_features=768, out_features=192, bias=True)
        self.dropout5 = Dropout(p=0.0, inplace=False)
        self.layernorm8 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.layernorm9 = LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        self.linear7 = Linear(in_features=192, out_features=768, bias=True)
        self.gelu3 = GELU(approximate='none')
        self.dropout6 = Dropout(p=0.0, inplace=False)
        self.linear8 = Linear(in_features=768, out_features=192, bias=True)
        self.dropout7 = Dropout(p=0.0, inplace=False)
        self.layernorm10 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear9 = Linear(in_features=768, out_features=384, bias=False)
        self.layernorm11 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm12 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu4 = GELU(approximate='none')
        self.dropout8 = Dropout(p=0.0, inplace=False)
        self.linear11 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout9 = Dropout(p=0.0, inplace=False)
        self.layernorm13 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm14 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.dropout10 = Dropout(p=0.0, inplace=False)
        self.linear13 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout11 = Dropout(p=0.0, inplace=False)
        self.layernorm15 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm16 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear14 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu6 = GELU(approximate='none')
        self.dropout12 = Dropout(p=0.0, inplace=False)
        self.linear15 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout13 = Dropout(p=0.0, inplace=False)
        self.layernorm17 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm18 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu7 = GELU(approximate='none')
        self.dropout14 = Dropout(p=0.0, inplace=False)
        self.linear17 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout15 = Dropout(p=0.0, inplace=False)
        self.layernorm19 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm20 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear18 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu8 = GELU(approximate='none')
        self.dropout16 = Dropout(p=0.0, inplace=False)
        self.linear19 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout17 = Dropout(p=0.0, inplace=False)
        self.layernorm21 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.layernorm22 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear20 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu9 = GELU(approximate='none')
        self.dropout18 = Dropout(p=0.0, inplace=False)
        self.linear21 = Linear(in_features=1536, out_features=384, bias=True)
        self.dropout19 = Dropout(p=0.0, inplace=False)
        self.layernorm23 = LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        self.linear22 = Linear(in_features=1536, out_features=768, bias=False)
        self.layernorm24 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.layernorm25 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear23 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu10 = GELU(approximate='none')
        self.dropout20 = Dropout(p=0.0, inplace=False)
        self.linear24 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout21 = Dropout(p=0.0, inplace=False)
        self.layernorm26 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.layernorm27 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear25 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu11 = GELU(approximate='none')
        self.dropout22 = Dropout(p=0.0, inplace=False)
        self.linear26 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout23 = Dropout(p=0.0, inplace=False)
        self.layernorm28 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=1)
        self.linear27 = Linear(in_features=768, out_features=1000, bias=True)
        self.relative_position_bias_table0 = torch.rand(torch.Size([169, 3])).to(torch.float32)
        self.relative_position_index0 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight0 = torch.rand(torch.Size([288, 96])).to(torch.float32)
        self.weight1 = torch.rand(torch.Size([96, 96])).to(torch.float32)
        self.bias0 = torch.rand(torch.Size([288])).to(torch.float32)
        self.bias1 = torch.rand(torch.Size([96])).to(torch.float32)
        self.relative_position_bias_table1 = torch.rand(torch.Size([169, 3])).to(torch.float32)
        self.relative_position_index1 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight2 = torch.rand(torch.Size([288, 96])).to(torch.float32)
        self.weight3 = torch.rand(torch.Size([96, 96])).to(torch.float32)
        self.bias2 = torch.rand(torch.Size([288])).to(torch.float32)
        self.bias3 = torch.rand(torch.Size([96])).to(torch.float32)
        self.relative_position_bias_table2 = torch.rand(torch.Size([169, 6])).to(torch.float32)
        self.relative_position_index2 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight4 = torch.rand(torch.Size([576, 192])).to(torch.float32)
        self.weight5 = torch.rand(torch.Size([192, 192])).to(torch.float32)
        self.bias4 = torch.rand(torch.Size([576])).to(torch.float32)
        self.bias5 = torch.rand(torch.Size([192])).to(torch.float32)
        self.relative_position_bias_table3 = torch.rand(torch.Size([169, 6])).to(torch.float32)
        self.relative_position_index3 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight6 = torch.rand(torch.Size([576, 192])).to(torch.float32)
        self.weight7 = torch.rand(torch.Size([192, 192])).to(torch.float32)
        self.bias6 = torch.rand(torch.Size([576])).to(torch.float32)
        self.bias7 = torch.rand(torch.Size([192])).to(torch.float32)
        self.relative_position_bias_table4 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index4 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight8 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight9 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias8 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias9 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table5 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index5 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight10 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight11 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias10 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias11 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table6 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index6 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight12 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight13 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias12 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias13 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table7 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index7 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight14 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight15 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias14 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias15 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table8 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index8 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight16 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight17 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias16 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias17 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table9 = torch.rand(torch.Size([169, 12])).to(torch.float32)
        self.relative_position_index9 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight18 = torch.rand(torch.Size([1152, 384])).to(torch.float32)
        self.weight19 = torch.rand(torch.Size([384, 384])).to(torch.float32)
        self.bias18 = torch.rand(torch.Size([1152])).to(torch.float32)
        self.bias19 = torch.rand(torch.Size([384])).to(torch.float32)
        self.relative_position_bias_table10 = torch.rand(torch.Size([169, 24])).to(torch.float32)
        self.relative_position_index10 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight20 = torch.rand(torch.Size([2304, 768])).to(torch.float32)
        self.weight21 = torch.rand(torch.Size([768, 768])).to(torch.float32)
        self.bias20 = torch.rand(torch.Size([2304])).to(torch.float32)
        self.bias21 = torch.rand(torch.Size([768])).to(torch.float32)
        self.relative_position_bias_table11 = torch.rand(torch.Size([169, 24])).to(torch.float32)
        self.relative_position_index11 = torch.rand(torch.Size([2401])).to(torch.int64)
        self.weight22 = torch.rand(torch.Size([2304, 768])).to(torch.float32)
        self.weight23 = torch.rand(torch.Size([768, 768])).to(torch.float32)
        self.bias22 = torch.rand(torch.Size([2304])).to(torch.float32)
        self.bias23 = torch.rand(torch.Size([768])).to(torch.float32)

    def forward(self, x):
        x0=x
        x1=self.conv2d0(x0)
        x2=torch.permute(x1, [0, 2, 3, 1])
        x3=self.layernorm0(x2)
        x4=self.layernorm1(x3)
        x7=operator.getitem(self.relative_position_bias_table0, self.relative_position_index0)
        x8=x7.view(49, 49, -1)
        x9=x8.permute(2, 0, 1)
        x10=x9.contiguous()
        x11=x10.unsqueeze(0)
        x16=torchvision.models.swin_transformer.shifted_window_attention(x4, self.weight0, self.weight1, x11, [7, 7], 3,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias0, proj_bias=self.bias1)
        x17=stochastic_depth(x16, 0.0, 'row', False)
        x18=operator.add(x3, x17)
        x19=self.layernorm2(x18)
        x20=self.linear0(x19)
        x21=self.gelu0(x20)
        x22=self.dropout0(x21)
        x23=self.linear1(x22)
        x24=self.dropout1(x23)
        x25=stochastic_depth(x24, 0.0, 'row', False)
        x26=operator.add(x18, x25)
        x27=self.layernorm3(x26)
        x30=operator.getitem(self.relative_position_bias_table1, self.relative_position_index1)
        x31=x30.view(49, 49, -1)
        x32=x31.permute(2, 0, 1)
        x33=x32.contiguous()
        x34=x33.unsqueeze(0)
        x39=torchvision.models.swin_transformer.shifted_window_attention(x27, self.weight2, self.weight3, x34, [7, 7], 3,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias2, proj_bias=self.bias3)
        x40=stochastic_depth(x39, 0.018181818181818184, 'row', False)
        x41=operator.add(x26, x40)
        x42=self.layernorm4(x41)
        x43=self.linear2(x42)
        x44=self.gelu1(x43)
        x45=self.dropout2(x44)
        x46=self.linear3(x45)
        x47=self.dropout3(x46)
        x48=stochastic_depth(x47, 0.018181818181818184, 'row', False)
        x49=operator.add(x41, x48)
        x50=builtins.getattr(x49, 'shape')
        x51=operator.getitem(x50, slice(-3, None, None))
        x52=operator.getitem(x51, 0)
        x53=operator.getitem(x51, 1)
        x54=operator.getitem(x51, 2)
        x55=operator.mod(x53, 2)
        x56=operator.mod(x52, 2)
        x57=torch.nn.functional.pad(x49, (0, 0, 0, x55, 0, x56))
        x58=operator.getitem(x57, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        x59=operator.getitem(x57, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        x60=operator.getitem(x57, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        x61=operator.getitem(x57, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        x62=torch.cat([x58, x59, x60, x61], -1)
        x63=self.layernorm5(x62)
        x64=self.linear4(x63)
        x65=self.layernorm6(x64)
        x68=operator.getitem(self.relative_position_bias_table2, self.relative_position_index2)
        x69=x68.view(49, 49, -1)
        x70=x69.permute(2, 0, 1)
        x71=x70.contiguous()
        x72=x71.unsqueeze(0)
        x77=torchvision.models.swin_transformer.shifted_window_attention(x65, self.weight4, self.weight5, x72, [7, 7], 6,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias4, proj_bias=self.bias5)
        x78=stochastic_depth(x77, 0.03636363636363637, 'row', False)
        x79=operator.add(x64, x78)
        x80=self.layernorm7(x79)
        x81=self.linear5(x80)
        x82=self.gelu2(x81)
        x83=self.dropout4(x82)
        x84=self.linear6(x83)
        x85=self.dropout5(x84)
        x86=stochastic_depth(x85, 0.03636363636363637, 'row', False)
        x87=operator.add(x79, x86)
        x88=self.layernorm8(x87)
        x91=operator.getitem(self.relative_position_bias_table3, self.relative_position_index3)
        x92=x91.view(49, 49, -1)
        x93=x92.permute(2, 0, 1)
        x94=x93.contiguous()
        x95=x94.unsqueeze(0)
        x100=torchvision.models.swin_transformer.shifted_window_attention(x88, self.weight6, self.weight7, x95, [7, 7], 6,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias6, proj_bias=self.bias7)
        x101=stochastic_depth(x100, 0.05454545454545456, 'row', False)
        x102=operator.add(x87, x101)
        x103=self.layernorm9(x102)
        x104=self.linear7(x103)
        x105=self.gelu3(x104)
        x106=self.dropout6(x105)
        x107=self.linear8(x106)
        x108=self.dropout7(x107)
        x109=stochastic_depth(x108, 0.05454545454545456, 'row', False)
        x110=operator.add(x102, x109)
        x111=builtins.getattr(x110, 'shape')
        x112=operator.getitem(x111, slice(-3, None, None))
        x113=operator.getitem(x112, 0)
        x114=operator.getitem(x112, 1)
        x115=operator.getitem(x112, 2)
        x116=operator.mod(x114, 2)
        x117=operator.mod(x113, 2)
        x118=torch.nn.functional.pad(x110, (0, 0, 0, x116, 0, x117))
        x119=operator.getitem(x118, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        x120=operator.getitem(x118, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        x121=operator.getitem(x118, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        x122=operator.getitem(x118, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        x123=torch.cat([x119, x120, x121, x122], -1)
        x124=self.layernorm10(x123)
        x125=self.linear9(x124)
        x126=self.layernorm11(x125)
        x129=operator.getitem(self.relative_position_bias_table4, self.relative_position_index4)
        x130=x129.view(49, 49, -1)
        x131=x130.permute(2, 0, 1)
        x132=x131.contiguous()
        x133=x132.unsqueeze(0)
        x138=torchvision.models.swin_transformer.shifted_window_attention(x126, self.weight8, self.weight9, x133, [7, 7], 12,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias8, proj_bias=self.bias9)
        x139=stochastic_depth(x138, 0.07272727272727274, 'row', False)
        x140=operator.add(x125, x139)
        x141=self.layernorm12(x140)
        x142=self.linear10(x141)
        x143=self.gelu4(x142)
        x144=self.dropout8(x143)
        x145=self.linear11(x144)
        x146=self.dropout9(x145)
        x147=stochastic_depth(x146, 0.07272727272727274, 'row', False)
        x148=operator.add(x140, x147)
        x149=self.layernorm13(x148)
        x152=operator.getitem(self.relative_position_bias_table5, self.relative_position_index5)
        x153=x152.view(49, 49, -1)
        x154=x153.permute(2, 0, 1)
        x155=x154.contiguous()
        x156=x155.unsqueeze(0)
        x161=torchvision.models.swin_transformer.shifted_window_attention(x149, self.weight10, self.weight11, x156, [7, 7], 12,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias10, proj_bias=self.bias11)
        x162=stochastic_depth(x161, 0.09090909090909091, 'row', False)
        x163=operator.add(x148, x162)
        x164=self.layernorm14(x163)
        x165=self.linear12(x164)
        x166=self.gelu5(x165)
        x167=self.dropout10(x166)
        x168=self.linear13(x167)
        x169=self.dropout11(x168)
        x170=stochastic_depth(x169, 0.09090909090909091, 'row', False)
        x171=operator.add(x163, x170)
        x172=self.layernorm15(x171)
        x175=operator.getitem(self.relative_position_bias_table6, self.relative_position_index6)
        x176=x175.view(49, 49, -1)
        x177=x176.permute(2, 0, 1)
        x178=x177.contiguous()
        x179=x178.unsqueeze(0)
        x184=torchvision.models.swin_transformer.shifted_window_attention(x172, self.weight12, self.weight13, x179, [7, 7], 12,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias12, proj_bias=self.bias13)
        x185=stochastic_depth(x184, 0.10909090909090911, 'row', False)
        x186=operator.add(x171, x185)
        x187=self.layernorm16(x186)
        x188=self.linear14(x187)
        x189=self.gelu6(x188)
        x190=self.dropout12(x189)
        x191=self.linear15(x190)
        x192=self.dropout13(x191)
        x193=stochastic_depth(x192, 0.10909090909090911, 'row', False)
        x194=operator.add(x186, x193)
        x195=self.layernorm17(x194)
        x198=operator.getitem(self.relative_position_bias_table7, self.relative_position_index7)
        x199=x198.view(49, 49, -1)
        x200=x199.permute(2, 0, 1)
        x201=x200.contiguous()
        x202=x201.unsqueeze(0)
        x207=torchvision.models.swin_transformer.shifted_window_attention(x195, self.weight14, self.weight15, x202, [7, 7], 12,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias14, proj_bias=self.bias15)
        x208=stochastic_depth(x207, 0.1272727272727273, 'row', False)
        x209=operator.add(x194, x208)
        x210=self.layernorm18(x209)
        x211=self.linear16(x210)
        x212=self.gelu7(x211)
        x213=self.dropout14(x212)
        x214=self.linear17(x213)
        x215=self.dropout15(x214)
        x216=stochastic_depth(x215, 0.1272727272727273, 'row', False)
        x217=operator.add(x209, x216)
        x218=self.layernorm19(x217)
        x221=operator.getitem(self.relative_position_bias_table8, self.relative_position_index8)
        x222=x221.view(49, 49, -1)
        x223=x222.permute(2, 0, 1)
        x224=x223.contiguous()
        x225=x224.unsqueeze(0)
        x230=torchvision.models.swin_transformer.shifted_window_attention(x218, self.weight16, self.weight17, x225, [7, 7], 12,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias16, proj_bias=self.bias17)
        x231=stochastic_depth(x230, 0.14545454545454548, 'row', False)
        x232=operator.add(x217, x231)
        x233=self.layernorm20(x232)
        x234=self.linear18(x233)
        x235=self.gelu8(x234)
        x236=self.dropout16(x235)
        x237=self.linear19(x236)
        x238=self.dropout17(x237)
        x239=stochastic_depth(x238, 0.14545454545454548, 'row', False)
        x240=operator.add(x232, x239)
        x241=self.layernorm21(x240)
        x244=operator.getitem(self.relative_position_bias_table9, self.relative_position_index9)
        x245=x244.view(49, 49, -1)
        x246=x245.permute(2, 0, 1)
        x247=x246.contiguous()
        x248=x247.unsqueeze(0)
        x253=torchvision.models.swin_transformer.shifted_window_attention(x241, self.weight18, self.weight19, x248, [7, 7], 12,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias18, proj_bias=self.bias19)
        x254=stochastic_depth(x253, 0.16363636363636364, 'row', False)
        x255=operator.add(x240, x254)
        x256=self.layernorm22(x255)
        x257=self.linear20(x256)
        x258=self.gelu9(x257)
        x259=self.dropout18(x258)
        x260=self.linear21(x259)
        x261=self.dropout19(x260)
        x262=stochastic_depth(x261, 0.16363636363636364, 'row', False)
        x263=operator.add(x255, x262)
        x264=builtins.getattr(x263, 'shape')
        x265=operator.getitem(x264, slice(-3, None, None))
        x266=operator.getitem(x265, 0)
        x267=operator.getitem(x265, 1)
        x268=operator.getitem(x265, 2)
        x269=operator.mod(x267, 2)
        x270=operator.mod(x266, 2)
        x271=torch.nn.functional.pad(x263, (0, 0, 0, x269, 0, x270))
        x272=operator.getitem(x271, (Ellipsis, slice(0, None, 2), slice(0, None, 2), slice(None, None, None)))
        x273=operator.getitem(x271, (Ellipsis, slice(1, None, 2), slice(0, None, 2), slice(None, None, None)))
        x274=operator.getitem(x271, (Ellipsis, slice(0, None, 2), slice(1, None, 2), slice(None, None, None)))
        x275=operator.getitem(x271, (Ellipsis, slice(1, None, 2), slice(1, None, 2), slice(None, None, None)))
        x276=torch.cat([x272, x273, x274, x275], -1)
        x277=self.layernorm23(x276)
        x278=self.linear22(x277)
        x279=self.layernorm24(x278)
        x282=operator.getitem(self.relative_position_bias_table10, self.relative_position_index10)
        x283=x282.view(49, 49, -1)
        x284=x283.permute(2, 0, 1)
        x285=x284.contiguous()
        x286=x285.unsqueeze(0)
        x291=torchvision.models.swin_transformer.shifted_window_attention(x279, self.weight20, self.weight21, x286, [7, 7], 24,shift_size=[0, 0], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias20, proj_bias=self.bias21)
        x292=stochastic_depth(x291, 0.18181818181818182, 'row', False)
        x293=operator.add(x278, x292)
        x294=self.layernorm25(x293)
        x295=self.linear23(x294)
        x296=self.gelu10(x295)
        x297=self.dropout20(x296)
        x298=self.linear24(x297)
        x299=self.dropout21(x298)
        x300=stochastic_depth(x299, 0.18181818181818182, 'row', False)
        x301=operator.add(x293, x300)
        x302=self.layernorm26(x301)
        x305=operator.getitem(self.relative_position_bias_table11, self.relative_position_index11)
        x306=x305.view(49, 49, -1)
        x307=x306.permute(2, 0, 1)
        x308=x307.contiguous()
        x309=x308.unsqueeze(0)
        x314=torchvision.models.swin_transformer.shifted_window_attention(x302, self.weight22, self.weight23, x309, [7, 7], 24,shift_size=[3, 3], attention_dropout=0.0, dropout=0.0, qkv_bias=self.bias22, proj_bias=self.bias23)
        x315=stochastic_depth(x314, 0.2, 'row', False)
        x316=operator.add(x301, x315)
        x317=self.layernorm27(x316)
        x318=self.linear25(x317)
        x319=self.gelu11(x318)
        x320=self.dropout22(x319)
        x321=self.linear26(x320)
        x322=self.dropout23(x321)
        x323=stochastic_depth(x322, 0.2, 'row', False)
        x324=operator.add(x316, x323)
        x325=self.layernorm28(x324)
        x326=x325.permute(0, 3, 1, 2)
        x327=self.adaptiveavgpool2d0(x326)
        x328=torch.flatten(x327, 1)
        x329=self.linear27(x328)

m = M().eval()
CORES=os.popen("lscpu | grep Core | awk '{print $4}'").readlines()
SOCKETS=os.popen("lscpu | grep Socket | awk '{print $2}'").readlines()
BS=int(CORES[0])*int(SOCKETS[0])
batch_size=BS
x = torch.rand(1, 3, 224, 224)
def print_throughput(flag):
    start_time=time.time()
    for i in range(10):
        output = m(x)
    total_iter_time = time.time() - start_time
    Throughput = batch_size * 10 / total_iter_time
    file_current = os.path.basename(__file__)
    print(file_current,',',BS,',',flag,',',Throughput)
for flag in {False,True}:
    torch._C._jit_set_texpr_fuser_enabled(flag)
    print_throughput(flag)
