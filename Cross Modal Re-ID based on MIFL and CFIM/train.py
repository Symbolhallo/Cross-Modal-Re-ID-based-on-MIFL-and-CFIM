from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net, modal_Classifier
from utils import *
from loss import OriTripletLoss, CenterTripletLoss, CrossEntropyLabelSmooth, TripletLoss_WRT
from tensorboardX import SummaryWriter
from re_rank import random_walk, k_reciprocal

import numpy as np

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=100, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=8, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--share_net', default=2, type=int, metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--cd', default='on', type=str, help='performing PCB, on or off')
parser.add_argument('--m3', default='on', type=str, help='modal, on or off')
parser.add_argument('--w_center', default=2.0, type=float, help='the weight for center loss')
parser.add_argument('--local_feat_dim', default=256, type=int, help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=6, type=int, help='num of local strips in PCB')
parser.add_argument('--label_smooth', default='off', type=str, help='performing label smooth or not')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/media/a/file5/symbol/data/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]
elif dataset == 'regdb':
    data_path = '/media/a/file5/symbol/data/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset + '_c_tri_cd_{}_w_tri_{}'.format(args.cd, args.w_center)
if args.cd == 'on':
    suffix = suffix + '_s{}_f{}'.format(args.num_strips, args.local_feat_dim)

suffix = suffix + '_share_net{}'.format(args.share_net)
if args.method == 'agw':
    suffix = suffix + '_agw_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_gm10_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_ac = 0
start_epoch = 0
W = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='on', arch=args.arch, share_net=args.share_net, cd=args.cd, m3=args.cd,
                    local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch, share_net=args.share_net, cd=args.cd)
net.to(device)
net_modal_classifier = modal_Classifier(embed_dim=2048, modal_class=3)
net_modal_classifier.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
if args.label_smooth == 'on':
    criterion_id = CrossEntropyLabelSmooth(n_class)
else:
    criterion_id = nn.CrossEntropyLoss()

if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
    criterion_tri = CenterTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)

if args.optim == 'sgd':
    if args.cd == 'on':
        ignored_params = list(map(id, net.local_conv_list1.parameters())) \
                         + list(map(id, net.local_conv_list2.parameters())) \
                         + list(map(id, net.local_conv_list3.parameters())) \
                         + list(map(id, net.local_conv_list4.parameters())) \
                         + list(map(id, net.fc_channel_list.parameters())) \
                         + list(map(id, net.modal_classifier.parameters())) \
                         + list(map(id, net.modal_bottleneck.parameters())) \
                         + list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier_4.parameters())) \
                         + list(map(id, net.fc_list1.parameters())) \
                         + list(map(id, net.fc_list2.parameters())) \
                         + list(map(id, net.fc_list3.parameters())) \
                         + list(map(id, net.fc_list4.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.classifier_4.parameters(), 'lr': args.lr},
            {'params': net.modal_bottleneck.parameters(), 'lr': args.lr},
            {'params': net.modal_classifier.parameters(), 'lr': args.lr},
            {'params': net.fc_channel_list.parameters(), 'lr': args.lr},
            {'params': net.local_conv_list1.parameters(), 'lr': args.lr},
            {'params': net.fc_list1.parameters(), 'lr': args.lr},
            {'params': net.local_conv_list2.parameters(), 'lr': args.lr},
            {'params': net.fc_list2.parameters(), 'lr': args.lr},
            {'params': net.local_conv_list3.parameters(), 'lr': args.lr},
            {'params': net.fc_list3.parameters(), 'lr': args.lr},
            {'params': net.local_conv_list4.parameters(), 'lr': args.lr},
            {'params': net.fc_list4.parameters(), 'lr': args.lr}
        ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 20:
        lr = args.lr
    elif 20 <= epoch < 50:
        lr = args.lr * 0.1
    elif 50 <= epoch < 60:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch, W):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)
        labels = labels.long()

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        # 构建划语义标签
        len_4 = len(label1)
        label_0 = torch.zeros(len_4 * 2).long()
        label_1 = torch.ones(len_4 * 2).long()
        label_2 = (torch.ones(len_4 * 2) * 2).long()
        label_3 = (torch.ones(len_4 * 2) * 3).long()
        label_4 = torch.cat((label_0, label_1, label_2, label_3), 0)
        label_4 = Variable(label_4.cuda())

        # 构建模态标签
        label_rgb_len = len(label1)
        label_ir_len = len(label2)
        modal_rgb_labels = torch.zeros(label_rgb_len).long()
        modal_ir_labels = torch.ones(label_ir_len).long()
        modal_mix_labels = (torch.ones(len(labels)) * 2).long()
        modal_labels = torch.cat((modal_rgb_labels, modal_ir_labels), 0)
        modal_labels = Variable(modal_labels.cuda())
        modal_mix_labels = Variable(modal_mix_labels.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        if args.cd == 'on':
            local_feat_list_all, logits_list_all, channel_scores_list, feat_list, feat_all, loss_reg, feat_x, modal_score, modal_feat, score_4 = net(input1, input2)

            # 通道分组的分类损失
            channel_cd_loss1 = criterion_id(channel_scores_list[0], labels)
            channel_cd_loss2 = criterion_id(channel_scores_list[1], labels)
            channel_cd_loss3 = criterion_id(channel_scores_list[2], labels)
            channel_cd_loss4 = criterion_id(channel_scores_list[3], labels)
            chaneel_cd_loss = channel_cd_loss1 + channel_cd_loss2 + channel_cd_loss3 + channel_cd_loss4
            # 语义分类
            cd_4 = criterion_id(score_4, label_4)

            # 模态3分类
            loss_modal = criterion_id(modal_score, modal_labels)
            loss_m_3 = criterion_id(net_modal_classifier(modal_feat), modal_labels)
            loss_x_3 = criterion_id(net_modal_classifier(feat_x), modal_mix_labels)

            # channel 1
            loss_id1 = criterion_id(logits_list_all[0][0], labels)
            loss_tri_l1, batch_acc = criterion_tri(local_feat_list_all[0][0], labels)
            for i in range(len(local_feat_list_all[0]) - 1):
                loss_id1 += criterion_id(logits_list_all[0][i + 1], labels)
                loss_tri_l1 += criterion_tri(local_feat_list_all[0][i + 1], labels)[0]
            loss_tri1, batch_acc = criterion_tri(feat_list[0], labels)
            loss_tri1 += loss_tri_l1 * args.w_center  #
            loss1 = loss_id1 + loss_tri1

            # channel 2
            loss_id2 = criterion_id(logits_list_all[1][0], labels)
            loss_tri_l2, batch_acc = criterion_tri(local_feat_list_all[1][0], labels)
            for i in range(len(local_feat_list_all[1]) - 1):
                loss_id2 += criterion_id(logits_list_all[1][i + 1], labels)
                loss_tri_l2 += criterion_tri(local_feat_list_all[1][i + 1], labels)[0]
            loss_tri2, batch_acc = criterion_tri(feat_list[1], labels)
            loss_tri2 += loss_tri_l2 * args.w_center  #
            loss2 = loss_id2 + loss_tri2

            # channel 3
            loss_id3 = criterion_id(logits_list_all[2][0], labels)
            loss_tri_l3, batch_acc = criterion_tri(local_feat_list_all[2][0], labels)
            for i in range(len(local_feat_list_all[2]) - 1):
                loss_id3 += criterion_id(logits_list_all[2][i + 1], labels)
                loss_tri_l3 += criterion_tri(local_feat_list_all[2][i + 1], labels)[0]
            loss_tri3, batch_acc = criterion_tri(feat_list[2], labels)
            loss_tri3 += loss_tri_l3 * args.w_center  #
            loss3 = loss_id3 + loss_tri3

            # channel 4
            loss_id4 = criterion_id(logits_list_all[3][0], labels)
            loss_tri_l4, batch_acc = criterion_tri(local_feat_list_all[3][0], labels)
            for i in range(len(local_feat_list_all[3]) - 1):
                loss_id4 += criterion_id(logits_list_all[3][i + 1], labels)
                loss_tri_l4 += criterion_tri(local_feat_list_all[3][i + 1], labels)[0]
            loss_tri4, batch_acc = criterion_tri(feat_list[3], labels)
            loss_tri4 += loss_tri_l4 * args.w_center  #
            loss4 = loss_id4 + loss_tri4

            loss_tri, batch_acc = criterion_tri(feat_all, labels)
            correct += batch_acc
            loss_id = loss_id1 + loss_id2 + loss_id3 + loss_id3

            loss = loss1 + loss2 + loss3 + loss4 + 0.3 * loss_tri + 2 * cd_4 + W * (loss_modal + loss_x_3 + loss_m_3) + chaneel_cd_loss  # 4 号机 61.67

        else:
            feat, out0 = net(input1, input2)
            loss_id = criterion_id(out0, labels)

            loss_tri, batch_acc = criterion_tri(feat, labels)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)
            loss = loss_id + loss_tri * args.w_center  #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri, 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    return 1. / (1. + train_loss.avg)



pool_dim = (args.num_strips * args.local_feat_dim) * 4
def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, test_mode[0])
            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, test_mode[1])
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc


def my_test(epoch):
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False,
                                            num_workers=args.workers)

        gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)
        # pool feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc_fc, mAP_fc, mINP_fc = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc_fc = cmc_fc
            all_mAP_fc = mAP_fc
            all_mINP_fc = mINP_fc

            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc_fc = all_cmc_fc + cmc_fc
            all_mAP_fc = all_mAP_fc + mAP_fc
            all_mINP_fc = all_mINP_fc + mINP_fc

            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))

    cmc_fc = all_cmc_fc / 10
    mAP_fc = all_mAP_fc / 10
    mINP_fc = all_mINP_fc / 10

    cmc_pool = all_cmc_pool / 10
    mAP_pool = all_mAP_pool / 10
    mINP_pool = all_mINP_pool / 10
    print('All Average:')
    print(
        'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
    print(
        'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
    writer.add_scalar('rank1_pool', cmc_pool[0], epoch)
    writer.add_scalar('mAP_pool', mAP_pool, epoch)
    writer.add_scalar('mINP_pool', mINP_pool, epoch)
    writer.add_scalar('rank1_fc', cmc_fc[0], epoch)
    writer.add_scalar('mAP_fc', mAP_fc, epoch)
    writer.add_scalar('mINP_fc', mINP_fc, epoch)
    return cmc_fc, mAP_fc, mINP_fc, cmc_pool, mAP_pool, mINP_pool


# training
print('==> Start Training...')
for epoch in range(start_epoch, 181 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    # print(trainset.cIndex)
    # print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch,
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    W = train(epoch, W)
    # if epoch > 0 and epoch % 2 == 0:
    if epoch >= 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        # cmc_fc, mAP_fc, mINP_fc, cmc_pool, mAP_pool, mINP_pool = my_test(epoch)
        cmc_fc, mAP_fc, mINP_fc = test(epoch)
        # save model
        if cmc_fc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_fc[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_fc,
                'mAP': mAP_fc,
                'mINP': mINP_fc,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_fc_best.t')

        print('当前这代的性能')
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))

        print('FC   Best Epoch [{}], Rank-1: {:.2%} '.format(best_epoch, best_acc))
########################################################################################################################