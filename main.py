import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.net import net
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
# from utils import *
from A import *
import random
import time
from net.losses import *
from utils.SSIM_loss import *
from utils.UIQM_loss import *
from net.frequency import frequency_loss


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Mamba-UIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=1e-5')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default=200, help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='./Dataset/UIE/UIEB/input')
parser.add_argument('--label_train', type=str, default='./Dataset/UIE/UIEB/label')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--data_test', type=str, default='./Dataset/UIE/UIEB/raw')
parser.add_argument('--label_test', type=str, default='./Dataset/UIE/UIEB/GT')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped HR image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='results/', help='Location to save checkpoint models')
parser.add_argument('--lambda_freq', type=float, default=0.05, help='weight of frequency loss')
parser.add_argument('--freq_hf_weight', type=float, default=0.5, help='high-frequency weight in frequency loss')


opt = parser.parse_args()


def seed_torch(seed=opt.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch()
cudnn.benchmark = True

mse_loss = torch.nn.MSELoss().cuda()
# color_loss = ColorLoss()
color_loss = ColorLossImproved()

# color_loss1 = ColorLoss1()


def train(epoch):
    epoch_loss = 0
    epoch_start = time.time()
    model.train()
    Edge = EdgeLoss()

    loss_1_sum = 0
    loss_3_sum = 0
    loss_ssim1_sum = 0
    loss_ssim2_sum = 0
    loss_edge_sum = 0
    loss_freq_sum = 0
    uiqm_sum = 0

    for batch in training_data_loader:
        input, label = batch[0], batch[1]

        input = input.cuda()
        label = label.cuda()

        j_out, t_out, tb_out = model(input)

        a_out = get_A(input).cuda()
        I_rec = j_out * t_out + (1 - tb_out) * a_out

        loss_1 = mse_loss(I_rec, input)
        loss_ssim1 = 1 - torch.mean(ssim(I_rec, input)).cuda()

        loss_3 = mse_loss(label, j_out)
        loss_ssim2 = 1 - torch.mean(ssim(label, j_out)).cuda()

        loss_edg = Edge(label, j_out)
        loss_freq = frequency_loss(j_out, label, hf_weight=opt.freq_hf_weight)

        uiqm_value = 0.0

        total_loss = loss_1 + loss_3 + loss_ssim1 + loss_ssim2 + 0.05 * loss_edg + opt.lambda_freq * loss_freq

        optimizer.zero_grad()
        total_loss.backward()
        epoch_loss += total_loss.item()
        loss_1_sum += loss_1.item()
        loss_3_sum += loss_3.item()
        loss_ssim1_sum += loss_ssim1.item()
        loss_ssim2_sum += loss_ssim2.item()
        loss_edge_sum += loss_edg.item()
        loss_freq_sum += loss_freq.item()
        uiqm_sum += uiqm_value
        optimizer.step()

    num_batches = len(training_data_loader)
    avg_loss = epoch_loss / num_batches
    epoch_time = time.time() - epoch_start
    print("===> Epoch[{}/{}]: Avg Loss: {:.4f} || rec_mse: {:.4f} || sup_mse: {:.4f} || rec_ssim: {:.4f} || sup_ssim: {:.4f} || edge: {:.4f} || freq: {:.4f} || uiqm: {:.4f} || Learning rate: lr={} || Timer: {:.2f} sec.".format(
        epoch, opt.nEpochs, avg_loss, loss_1_sum / num_batches, loss_3_sum / num_batches,
        loss_ssim1_sum / num_batches, loss_ssim2_sum / num_batches, loss_edge_sum / num_batches,
        loss_freq_sum / num_batches, uiqm_sum / num_batches, optimizer.param_groups[0]['lr'], epoch_time))
    return avg_loss

def checkpoint():
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    model_out_path = opt.save_folder + "best.pth"
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('===> Loading datasets')

    test_set = get_eval_set(opt.data_test, opt.label_test)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    train_set = get_training_set(opt.data_train, opt.label_train, opt.patch_size, opt.data_augmentation)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print('===> Building model ')

    model = net().cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    milestones = []
    for i in range(1, opt.nEpochs+1):
        if i % opt.decay == 0:
            milestones.append(i)


    scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

    for epoch in range(opt.start_iter, opt.nEpochs + 1):

        train(epoch)
        scheduler.step()

    checkpoint()
