import argparse
import csv
import  torch, os
import  numpy as np
from torchsummary import summary
from PIL import Image
from    MiniImagenet import MiniImagenet

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from meta import Meta

def To_Convert(x):
    return Image.open(x).convert('RGB')

def loadCSV(self, csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]
            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels

def main():
    # 设置随机种子
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    # args在ifmain里进行设置
    print(args)

    # T 加载图像并对图像进行预处理
    image_path_sup = r""
    image_path_que = r""
    csv_path_sup = r""
    csvdata_sup = loadCSV(csv_path_sup)

    image_data = []
    img2label = {}
    transform = transforms.Compose([To_Convert,
                                    transforms.Resize((args.imgsz, args.imgsz)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    for i, (k, v) in enumerate(csvdata_sup.items()):
        image_data.append(v)  # [[img1, img2, ...], [img111, ...]]
        img2label[k] = i  # {"img_name[:9]":label}
    cls_num = len(image_data)

    create_batch(args.batchsz)



    # 设置网络模型结构
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
        ('linear', [args.n_way, 4096])
    ]
    # 得到网络：maml框架
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

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 10 == 0:
                print('step:', step, '\ttraining acc:', accs)

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

        torch.save(maml.state_dict(), "H:/MAML-Pytorch-master/Run_Xu/15way_epoch{}_model.pth".format(epoch))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100000)#60000
    argparser.add_argument('--n_way', type=int, help='n way', default=15)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)   # 1
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)    # 15
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=227)   # 28或84
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)  # 1或3
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)# 4
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-4)#1e-3
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.005)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    device = torch.device("cuda:0")
    main()
