from __future__ import print_function

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from A import my_save_image, torch_to_np
from data import get_eval_set
from net.dpmamba import DPMambaUIE
from utils.ssm_psnr_utils import getPSNR, getSSIM


parser = argparse.ArgumentParser(description='PyTorch DPMamba-UIE ablation evaluation')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of data loader workers')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='./Dataset/UIE/UIEB/raw')
parser.add_argument('--label_test', type=str, default='./Dataset/UIE/UIEB/GT')
parser.add_argument('--model', type=str, required=True, help='checkpoint path')
parser.add_argument('--output_folder', type=str, default='./results_dpmamba/')
parser.add_argument('--save_aux_outputs', type=int, default=1, help='save A/T/TB/I_rec and aux outputs')
parser.add_argument('--override_config', type=int, default=0, help='use CLI switches instead of checkpoint config')
parser.add_argument('--legacy_state_dict', type=int, default=0, help='load old pure state_dict with strict=False')

# model switches, used when checkpoint has no config or override_config=1
parser.add_argument('--use_dynamic_mamba', type=int, default=0)
parser.add_argument('--use_learnable_physics', type=int, default=0)
parser.add_argument('--use_frequency_module', type=int, default=0)
parser.add_argument('--use_lightweight', type=int, default=0)
parser.add_argument('--use_a_residual_map', type=int, default=0)
parser.add_argument('--a_residual_scale', type=float, default=0.1)
parser.add_argument('--light_base_channels', type=int, default=32)

opt = parser.parse_args()


def get_cfg(config, key):
    return getattr(opt, key) if (config is None or opt.override_config) else config.get(key, getattr(opt, key))


def build_model(config, device):
    model = DPMambaUIE(
        use_dynamic_mamba=bool(int(get_cfg(config, 'use_dynamic_mamba'))),
        use_learnable_physics=bool(int(get_cfg(config, 'use_learnable_physics'))),
        use_frequency_module=bool(int(get_cfg(config, 'use_frequency_module'))),
        use_lightweight=bool(int(get_cfg(config, 'use_lightweight'))),
        use_a_residual_map=bool(int(get_cfg(config, 'use_a_residual_map'))),
        a_residual_scale=float(get_cfg(config, 'a_residual_scale')),
        light_base_channels=int(get_cfg(config, 'light_base_channels')),
    )
    return model.to(device)


def save_scan_weights(name, aux, output_folder):
    scan_weights = aux.get('scan_weights')
    if scan_weights is None:
        return
    scan_dir = os.path.join(output_folder, 'ScanWeights')
    os.makedirs(scan_dir, exist_ok=True)
    packed = {}
    for key, value in scan_weights.items():
        if value is not None:
            packed[key] = value.detach().cpu().numpy()[0, :, :, 0, 0]
    if packed:
        np.save(os.path.join(scan_dir, os.path.splitext(name)[0] + '.npy'), packed, allow_pickle=True)


def eval_model():
    print('===> Loading datasets')
    test_set = get_eval_set(opt.data_test, opt.label_test)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    if opt.gpu_mode and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('===> Loading checkpoint')
    checkpoint = torch.load(opt.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        config = checkpoint.get('config')
    else:
        state_dict = checkpoint
        config = None

    print('===> Building model')
    model = build_model(config, device)
    if opt.legacy_state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print('Legacy load with strict=False.')
        print('Missing keys:', missing)
        print('Unexpected keys:', unexpected)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    print('Pre-trained model is loaded.')

    output_j = os.path.join(opt.output_folder, 'J') + os.sep
    output_a = os.path.join(opt.output_folder, 'A') + os.sep
    output_t = os.path.join(opt.output_folder, 'T') + os.sep
    output_r = os.path.join(opt.output_folder, 'R') + os.sep
    output_tb = os.path.join(opt.output_folder, 'TB') + os.sep
    os.makedirs(output_j, exist_ok=True)
    if opt.save_aux_outputs:
        for output_dir in [output_a, output_t, output_r, output_tb]:
            os.makedirs(output_dir, exist_ok=True)

    psnr_list = []
    ssim_list = []

    print('\nEvaluation:')
    with torch.no_grad():
        for batch in testing_data_loader:
            input, label, name = batch[0], batch[1], batch[2]
            print(name)
            input = input.to(device)
            label = label.to(device)

            J, t, tb, A, I_rec, aux = model(input)

            j_out_np = np.clip(torch_to_np(J), 0, 1)
            label_np = np.clip(torch_to_np(label), 0, 1)

            j_out_img = (j_out_np.transpose(1, 2, 0) * 255).round().astype(np.uint8)
            label_img = (label_np.transpose(1, 2, 0) * 255).round().astype(np.uint8)
            psnr = getPSNR(j_out_img, label_img)
            ssim_value = getSSIM(j_out_img, label_img)
            psnr_list.append(psnr)
            ssim_list.append(ssim_value)
            print('{} PSNR: {:.4f} | SSIM: {:.4f}'.format(name[0], psnr, ssim_value))

            my_save_image(name[0], j_out_np, output_j)
            if opt.save_aux_outputs:
                t_out_np = np.clip(torch_to_np(t), 0, 1)
                a_out_np = np.clip(torch_to_np(A), 0, 1)
                I_rec_np = np.clip(torch_to_np(I_rec), 0, 1)
                tb_out_np = np.clip(torch_to_np(tb), 0, 1)
                my_save_image(name[0], t_out_np, output_t)
                my_save_image(name[0], a_out_np, output_a)
                my_save_image(name[0], I_rec_np, output_r)
                my_save_image(name[0], tb_out_np, output_tb)
                save_scan_weights(name[0], aux, opt.output_folder)

    if psnr_list:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        print('\nEvaluation metrics on test set:')
        print('Average PSNR: {:.4f}'.format(avg_psnr))
        print('Average SSIM: {:.4f}'.format(avg_ssim))
    else:
        print('\nNo test images were evaluated.')


if __name__ == '__main__':
    eval_model()
