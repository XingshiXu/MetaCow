import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from learner import Learner
from copy import deepcopy

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()
        # 接收args
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        # 接收网络模型结构config和参数args，构建实例化的Learner：：：net
        self.net = Learner(config, args.imgc, args.imgsz)
        # 构建元学习优化器meta_optim
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay=0)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        losses_q1 = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        losses_q2 = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits,z,z_auto  = self.net(x_spt[i], vars=None, bn_training=True)
            # print("@@@@")
            # print(logits.size())
            # print(y_spt[i].size())

            loss1 = F.cross_entropy(logits, y_spt[i])
            loss2 = F.mse_loss(z,z_auto)
            # print("loss_mrtaatrain_CE{}".format(loss1))
            # print("loss_mrtaatrain_auto{}".format(loss2))
            # print("loss_mrtaatrain{}".format(loss1 + loss2))
            grad = torch.autograd.grad(loss1+loss2, self.net.parameters(), allow_unused=True)#allow_unused=True是xxs加的
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q,z,z_auto = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q1 = F.cross_entropy(logits_q, y_qry[i])
                loss_q2 = F.mse_loss(z, z_auto)
                loss_q = loss_q1 + loss_q2

                losses_q[0] += loss_q
                losses_q1[0] += loss_q1
                losses_q2[0] += loss_q2

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, z,z_auto= self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q = F.cross_entropy(logits_q, y_qry[i])
                loss_q1 = F.cross_entropy(logits_q, y_qry[i])
                loss_q2 = F.mse_loss(z, z_auto)
                loss_q = loss_q1 + loss_q2
                losses_q[1] += loss_q

                losses_q1[1] += loss_q1
                losses_q2[1] += loss_q2

                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits,z,z_auto = self.net(x_spt[i], fast_weights, bn_training=True)
                loss1 = F.cross_entropy(logits, y_spt[i])
                loss2 = F.mse_loss(z, z_auto)
                loss = loss1 + loss2
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q,z,z_auto = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss1 = F.cross_entropy(logits_q, y_qry[i])

                loss2 = F.mse_loss(z, z_auto)
                loss = loss1 + loss2
                losses_q[k + 1] += loss

                losses_q1[k + 1] += loss1
                losses_q2[k + 1] += loss2

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        loss_q1 = losses_q1[-1] / task_num
        loss_q2 = losses_q2[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs,loss_q,loss_q1,loss_q2


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        # logits = net(x_spt)
        # loss = F.cross_entropy(logits, y_spt)
        # grad = torch.autograd.grad(loss, net.parameters())
        # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        logits,z,z_auto = net(x_spt)

        loss1 = F.cross_entropy(logits, y_spt)
        loss2 = F.mse_loss(z, z_auto)

        # loss_autoencoder = F.cross_entropy(Learner.z_old,)
        grad = torch.autograd.grad(loss1+loss2, net.parameters(),allow_unused=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))


        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q ,z,z_auto= net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q,z,z_auto = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits,z,z_auto = net(x_spt, fast_weights, bn_training=True)
            loss1 = F.cross_entropy(logits, y_spt)
            loss2 = F.mse_loss(z, z_auto)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss1+loss2, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q,z,z_auto = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
