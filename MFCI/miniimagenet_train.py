import  torch, os
import  numpy as np
from torchsummary import summary

from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from torch.utils.tensorboard import SummaryWriter

from meta import Meta
xxs_writer = SummaryWriter(log_dir="H:\MAML-Pytorch-master\Run_20220902way")

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    # 设置随机种子
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)
    np.random.seed(2022)
    # args在ifmain里进行设置
    print(args)
    # 设置网络模型结构
    # config = [
    #     ('conv2d', [32, 3, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 1, 0]),
    #     ('flatten', []),
    #     ('linear', [args.n_way, 32 * 5 * 5])
    # ]
    config = [
        ('conv2d', [96, 3, 11, 11, 4, 0]),
        ('relu', [True]),
        ('bn', [96]),
        ('max_pool2d', [3, 2, 0]),

        ('conv2d', [256, 96, 5, 5, 1, 2]),
        ('relu', [True]),
        ('bn', [256]),
        ('max_pool2d', [3, 2, 0]),

        ('conv2d', [384, 256, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [384]),

        ('conv2d', [384, 384, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [384]),

        ('conv2d', [256, 384, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [256]),
        ('max_pool2d', [3, 2, 0]),

        ('flatten', []),
        ('linear', [4096, 256 * 6 * 6]),
        ('relu', [True]),
        ('linear', [args.n_way, 4096]),

        ('linear', [512, 4096]),
        ('linear', [4096, 512]),
        ('relu', [True])




    ]
    # 得到网络：maml
    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    # summary(maml.net, (3, 227, 227),batch_size= 1,device='cuda')

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('mini-imagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('mini-imagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True, )
        print("db is OK")
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):


            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs,loss,loss_q_CE,loss_q_auto = maml(x_spt, y_spt, x_qry, y_qry)
            if (step % 50 == 0) & (step != 0):
                xxs_writer.add_scalar("loss", loss.item(), step + 10000 * epoch)
                xxs_writer.add_scalar("loss_CE", loss_q_CE.item(), step + 10000 * epoch)
                xxs_writer.add_scalar("loss_Autoencoder", loss_q_auto.item(), step + 10000 * epoch)
                xxs_writer.add_scalars("Loss", {"Total loss": loss.item(), "loss_CE": loss_q_CE.item(), "loss_Autoencoder": loss_q_auto.item()},
                                       step + 10000 * epoch)
                xxs_writer.add_scalar("train acc", accs[1], step + 10000 * epoch)
                xxs_writer.add_scalars("train accs", {"frist": accs[1], "second": accs[2], "fifth": accs[5]},
                                       step + 10000 * epoch)

            if step % 15 == 0:
                print('step:', step, '\ttraining acc:', accs,"\tTotal loss", loss.item(), "\tloss_CE", loss_q_CE.item(),"\tloss_Autoencoder", loss_q_auto.item())
                print("--"*50)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)
                xxs_writer.add_scalar("val acc", accs[1], step+10000*epoch)
                xxs_writer.add_scalars("val accs", {"frist":accs[1], "second":accs[2], "fifth":accs[5]},step+10000*epoch)
        torch.save(maml.state_dict(), "H:/MAML-Pytorch-master/Run_Xu/20220902way_epoch{}_model.pth".format(epoch))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50000)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)   # 1
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=1)    # 15
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=227)   # 28或84
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)  # 1或3
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)# 4
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=2e-4)#1e-3
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
