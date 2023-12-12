import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np


# Base-leaner
class Learner(nn.Module):
    """

    """
    z_old = None  # xxs
    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)网络配置文件
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config
        # this dict contains all tensors needed to be optimized
        # 这个字典包含所有需要优化的张量
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        # 进行Kaiming初始化
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)# 使用正态分布对输入张量进行赋值
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))


            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError


    # 得到网络模型结构，设置模块的额外表示信息
    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def forward(self, x, vars=None, bn_training=True):
        """
        这个函数可以通过finetunning调用，但是在finetunning中，我们不希望更新running_mean/running_var。
        考虑到 bn层中 weights/bias被更新，它已经被 fast_weights 分开。
        实际上，为了不更新running_mean/running_var，我们需要设置 update_bn_statistics=False
        但是 weight/bias 通过 fast_weiths 被更新和 不dirty初始化 参数θ。

        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        # Alexnet
        if vars is None:
            vars = self.vars
        x = F.conv2d(x, weight=vars[0], bias=vars[1], stride=4, padding=0)
        x = F.relu(x, inplace=True)
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=vars[2], bias=vars[3], training=bn_training)
        x = F.max_pool2d(x, 3, 2, 0)

        x = F.conv2d(x, weight=vars[4], bias=vars[5], stride=1, padding=2)
        x = F.relu(x, inplace=True)
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=vars[6], bias=vars[7], training=bn_training)
        x = F.max_pool2d(x, 3, 2, 0)

        x = F.conv2d(x, weight=vars[8], bias=vars[9], stride=1, padding=1)
        x = F.relu(x, inplace=True)
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        x = F.batch_norm(x, running_mean, running_var, weight=vars[10], bias=vars[11], training=bn_training)

        x = F.conv2d(x, weight=vars[12], bias=vars[13], stride=1, padding=1)
        x = F.relu(x, inplace=True)
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        x = F.batch_norm(x, running_mean, running_var, weight=vars[14], bias=vars[15], training=bn_training)

        x = F.conv2d(x, weight=vars[16], bias=vars[17], stride=1, padding=1)
        x = F.relu(x, inplace=True)
        running_mean, running_var = self.vars_bn[8], self.vars_bn[9]
        x = F.batch_norm(x, running_mean, running_var, weight=vars[18], bias=vars[19], training=bn_training)
        x = F.max_pool2d(x, 3, 2, 0)

        # print("@@@@{}".format(z))
        x = x.view(x.size(0), -1) # flatten

        # print(z_old.size())
        x = F.linear(x, vars[20], vars[21])
        z = x
        z_old = z
        #z_old = z.view(1, -1)

        x = F.relu(x, inplace=True)
        x = F.linear(x, vars[22], vars[23])

        # autoencoder
        z = F.linear(z, vars[24], vars[25])

        z_auto = F.linear(z, vars[26], vars[27])
        z_auto = F.relu(z_auto, inplace=True)

        #z_auto = z_auto.view(1, -1)

        # print(z_auto.size())
        return x,z_old,z_auto
        # return x,x,x

        # if vars is None:
        #     vars = self.vars
        #
        # idx = 0
        # bn_idx = 0

        # for name, param in self.config:
        #     if name is 'conv2d':
        #         w, b = vars[idx], vars[idx + 1]
        #         # remember to keep synchrozied of forward_encoder and forward_decoder!
        #         x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
        #         idx += 2
        #         # print(name, param, '\tout:', x.shape)
        #     elif name is 'convt2d':
        #         w, b = vars[idx], vars[idx + 1]
        #         # remember to keep synchrozied of forward_encoder and forward_decoder!
        #         x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
        #         idx += 2
        #         # print(name, param, '\tout:', x.shape)
        #     elif name is 'linear':
        #         w, b = vars[idx], vars[idx + 1]
        #         x = F.linear(x, w, b)
        #         idx += 2
        #         # print('forward:', idx, x.norm().item())
        #     elif name is 'bn':
        #         w, b = vars[idx], vars[idx + 1]
        #         running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
        #         x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
        #         idx += 2
        #         bn_idx += 2
        #
        #     elif name is 'flatten':
        #         # print(x.shape)
        #         x = x.view(x.size(0), -1)
        #     elif name is 'reshape':
        #         # [b, 8] => [b, 2, 2, 2]
        #         x = x.view(x.size(0), *param)
        #     elif name is 'relu':
        #         x = F.relu(x, inplace=param[0])
        #     elif name is 'leakyrelu':
        #         x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
        #     elif name is 'tanh':
        #         x = F.tanh(x)
        #     elif name is 'sigmoid':
        #         x = torch.sigmoid(x)
        #     elif name is 'upsample':
        #         x = F.upsample_nearest(x, scale_factor=param[0])
        #     elif name is 'max_pool2d':
        #         x = F.max_pool2d(x, param[0], param[1], param[2])
        #     elif name is 'avg_pool2d':
        #         x = F.avg_pool2d(x, param[0], param[1], param[2])
        #
        #     else:
        #         raise NotImplementedError
        #
        # # make sure variable is used properly
        # assert idx == len(vars)
        # assert bn_idx == len(self.vars_bn)
        #
        #
        # return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars