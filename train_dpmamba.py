import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader

from data import get_training_set
from net.dpmamba import DPMambaUIE
from net.frequency import frequency_loss
from net.losses import ColorLossImproved, EdgeLoss
from net.physics import smoothness_loss
from utils.SSIM_loss import ssim


parser = argparse.ArgumentParser(description='PyTorch DPMamba-UIE ablation training')

# model switches
parser.add_argument('--use_dynamic_mamba', type=int, default=0, help='enable dynamic Mamba fusion')
parser.add_argument('--use_learnable_physics', type=int, default=0, help='enable learnable physical parameter A')
parser.add_argument('--use_frequency_module', type=int, default=0, help='enable FCTC module')
parser.add_argument('--use_lightweight', type=int, default=0, help='enable lightweight physical branch')
parser.add_argument('--use_a_residual_map', type=int, default=0, help='enable residual A map for learnable/lightweight physics')
parser.add_argument('--a_residual_scale', type=float, default=0.1, help='scale of residual A map')
parser.add_argument('--light_base_channels', type=int, default=32, help='base channels for lightweight physical branch')

# loss switches
parser.add_argument('--use_freq_loss', type=int, default=0, help='enable frequency loss')
parser.add_argument('--use_smooth_loss', type=int, default=0, help='enable t/tb smoothness loss')
parser.add_argument('--use_color_loss', type=int, default=0, help='enable color loss')

# loss weights
parser.add_argument('--lambda_rec', type=float, default=1.0, help='weight of physical reconstruction MSE loss')
parser.add_argument('--lambda_sup', type=float, default=1.0, help='weight of supervised MSE loss')
parser.add_argument('--lambda_rec_ssim', type=float, default=1.0, help='weight of physical reconstruction SSIM loss')
parser.add_argument('--lambda_sup_ssim', type=float, default=1.0, help='weight of supervised SSIM loss')
parser.add_argument('--lambda_edge', type=float, default=0.05, help='weight of edge loss')
parser.add_argument('--lambda_freq', type=float, default=0.05, help='weight of frequency loss')
parser.add_argument('--freq_hf_weight', type=float, default=0.5, help='high-frequency weight in frequency loss')
parser.add_argument('--lambda_smooth', type=float, default=0.01, help='weight of smoothness loss')
parser.add_argument('--lambda_color', type=float, default=0.01, help='weight of color loss')

# experiment management
parser.add_argument('--experiment_name', type=str, default='debug', help='experiment folder name')
parser.add_argument('--save_root', type=str, default='./experiments', help='root folder for experiments')
parser.add_argument('--save_aux_outputs', type=int, default=0, help='reserved for compatibility with eval config')
parser.add_argument('--resume', type=str, default='', help='path to checkpoint for resuming training')
parser.add_argument('--resume_optimizer', type=int, default=1, help='resume optimizer state if available')
parser.add_argument('--resume_scheduler', type=int, default=1, help='resume scheduler state if available')

# training settings
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of data loader workers')
parser.add_argument('--decay', type=int, default=200, help='step scheduler decay interval')
parser.add_argument('--gamma', type=float, default=0.5, help='step scheduler decay factor')
parser.add_argument('--use_cosine_scheduler', type=int, default=0, help='enable cosine annealing scheduler')
parser.add_argument('--eta_min', type=float, default=1e-7, help='minimum LR for cosine scheduler')
parser.add_argument('--use_amp', type=int, default=0, help='enable AMP mixed precision training')
parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'], help='AMP dtype: fp16 is faster but bf16 is more numerically stable')
parser.add_argument('--use_tf32', type=int, default=0, help='enable TF32 acceleration on supported NVIDIA GPUs')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--data_train', type=str, default='./Dataset/UIE/UIEB/input')
parser.add_argument('--label_train', type=str, default='./Dataset/UIE/UIEB/label')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--rgb_range', type=int, default=1)

opt = parser.parse_args()


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_config(exp_dir):
    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(opt), f, indent=2, ensure_ascii=False)


def save_checkpoint(exp_dir, model, optimizer, scheduler, epoch, avg_loss, best_loss, name):
    ckpt = {
        'model_version': 'dpmamba_v1',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'config': vars(opt),
        'epoch': epoch,
        'avg_loss': avg_loss,
        'best_loss': best_loss,
    }
    torch.save(ckpt, os.path.join(exp_dir, name))


def build_model(device):
    model = DPMambaUIE(
        use_dynamic_mamba=bool(opt.use_dynamic_mamba),
        use_learnable_physics=bool(opt.use_learnable_physics),
        use_frequency_module=bool(opt.use_frequency_module),
        use_lightweight=bool(opt.use_lightweight),
        use_a_residual_map=bool(opt.use_a_residual_map),
        a_residual_scale=opt.a_residual_scale,
        light_base_channels=opt.light_base_channels,
    )
    return model.to(device)


def resume_training(model, optimizer, scheduler, device):
    if not opt.resume:
        return float('inf')

    if not os.path.isfile(opt.resume):
        raise FileNotFoundError('Resume checkpoint not found: {}'.format(opt.resume))

    checkpoint = torch.load(opt.resume, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])

        if opt.resume_optimizer and 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if opt.resume_scheduler and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

        resume_epoch = int(checkpoint.get('epoch', 0))
        if opt.start_iter <= resume_epoch:
            opt.start_iter = resume_epoch + 1

        best_loss = float(checkpoint.get('best_loss', checkpoint.get('avg_loss', float('inf'))))
        print('Resumed from {}, start epoch {}'.format(opt.resume, opt.start_iter))
        return best_loss

    model.load_state_dict(checkpoint)
    print('Loaded pure state_dict from {}. Optimizer and epoch are not resumed.'.format(opt.resume))
    return float('inf')


def train(epoch):
    model.train()
    epoch_start = time.time()
    epoch_loss = 0.0

    sums = {
        'rec_mse': 0.0,
        'sup_mse': 0.0,
        'rec_ssim': 0.0,
        'sup_ssim': 0.0,
        'edge': 0.0,
        'freq': 0.0,
        'smooth': 0.0,
        'color': 0.0,
    }

    use_amp = bool(opt.use_amp) and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if opt.amp_dtype == 'bf16' else torch.float16

    for batch in training_data_loader:
        input, label = batch[0].to(device), batch[1].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            J, t, tb, A, I_rec, aux = model(input)

        # Keep the model forward in mixed precision, but compute losses in FP32.
        # SSIM_loss.py requires both inputs to have the same dtype, and FP32 losses
        # are more stable for PSNR-oriented training.
        J_loss = J.float()
        t_loss = t.float()
        tb_loss = tb.float()
        I_rec_loss = I_rec.float()
        input_loss = input.float()
        label_loss = label.float()

        loss_rec_mse = mse_loss(I_rec_loss, input_loss)
        loss_rec_ssim = 1 - torch.mean(ssim(I_rec_loss, input_loss)).to(device)
        loss_sup_mse = mse_loss(label_loss, J_loss)
        loss_sup_ssim = 1 - torch.mean(ssim(label_loss, J_loss)).to(device)
        loss_edge = edge_loss(label_loss, J_loss)

        if opt.use_freq_loss:
            loss_freq = frequency_loss(J_loss, label_loss, hf_weight=opt.freq_hf_weight)
        else:
            loss_freq = J_loss.new_tensor(0.0)

        if opt.use_smooth_loss:
            loss_smooth = smoothness_loss(t_loss, tb_loss)
        else:
            loss_smooth = J_loss.new_tensor(0.0)

        if opt.use_color_loss:
            loss_color = torch.mean(color_loss(J_loss))
        else:
            loss_color = J_loss.new_tensor(0.0)

        total_loss = (
            opt.lambda_rec * loss_rec_mse +
            opt.lambda_rec_ssim * loss_rec_ssim +
            opt.lambda_sup * loss_sup_mse +
            opt.lambda_sup_ssim * loss_sup_ssim +
            opt.lambda_edge * loss_edge +
            opt.lambda_freq * loss_freq +
            opt.lambda_smooth * loss_smooth +
            opt.lambda_color * loss_color
        )

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        epoch_loss += total_loss.item()
        sums['rec_mse'] += loss_rec_mse.item()
        sums['sup_mse'] += loss_sup_mse.item()
        sums['rec_ssim'] += loss_rec_ssim.item()
        sums['sup_ssim'] += loss_sup_ssim.item()
        sums['edge'] += loss_edge.item()
        sums['freq'] += loss_freq.item()
        sums['smooth'] += loss_smooth.item()
        sums['color'] += loss_color.item()

    num_batches = len(training_data_loader)
    avg_loss = epoch_loss / num_batches
    epoch_time = time.time() - epoch_start
    msg = (
        '===> Epoch[{}/{}]: Avg Loss: {:.4f} || rec_mse: {:.4f} || sup_mse: {:.4f} || '
        'rec_ssim: {:.4f} || sup_ssim: {:.4f} || edge: {:.4f} || freq: {:.4f} || '
        'smooth: {:.4f} || color: {:.4f} || Learning rate: lr={} || Timer: {:.2f} sec.'
    ).format(
        epoch, opt.nEpochs, avg_loss,
        sums['rec_mse'] / num_batches,
        sums['sup_mse'] / num_batches,
        sums['rec_ssim'] / num_batches,
        sums['sup_ssim'] / num_batches,
        sums['edge'] / num_batches,
        sums['freq'] / num_batches,
        sums['smooth'] / num_batches,
        sums['color'] / num_batches,
        optimizer.param_groups[0]['lr'],
        epoch_time,
    )
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    return avg_loss


if __name__ == '__main__':
    seed_torch(opt.seed)
    cudnn.benchmark = True

    if bool(opt.use_tf32) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass

    if opt.gpu_mode and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    exp_dir = os.path.join(opt.save_root, opt.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_config(exp_dir)
    log_path = os.path.join(exp_dir, 'train.log')

    print('===> Loading datasets')
    train_set = get_training_set(opt.data_train, opt.label_train, opt.patch_size, opt.data_augmentation)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print('===> Building model')
    model = build_model(device)
    mse_loss = torch.nn.MSELoss().to(device)
    edge_loss = EdgeLoss()
    edge_loss.kernel = edge_loss.kernel.to(device)
    color_loss = ColorLossImproved().to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(opt.use_amp) and device.type == 'cuda')
    if opt.use_cosine_scheduler:
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=opt.nEpochs, eta_min=opt.eta_min)
    else:
        milestones = [i for i in range(1, opt.nEpochs + 1) if i % opt.decay == 0]
        scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

    best_loss = resume_training(model, optimizer, scheduler, device)
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        avg_loss = train(epoch)
        scheduler.step()
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(exp_dir, model, optimizer, scheduler, epoch, avg_loss, best_loss, 'best.pth')
            print('Checkpoint saved to {}'.format(os.path.join(exp_dir, 'best.pth')))
        save_checkpoint(exp_dir, model, optimizer, scheduler, epoch, avg_loss, best_loss, 'last.pth')
