from __future__ import print_function
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.net import net
from data import get_eval_set
from utils.ssm_psnr_utils import getPSNR, getSSIM
# from utils import *
from A import *
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch Mamba-UIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='./Dataset/UIE/UIEB/raw')
parser.add_argument('--label_test', type=str, default='./Dataset/UIE/UIEB/GT')
parser.add_argument('--model', default='weights/best.pth', help='Pretrained base model')
parser.add_argument('--output_folder', type=str, default='./results/')

opt = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for eval.py. Please run this script on a GPU environment.')

print('===> Loading datasets')
test_set = get_eval_set(opt.data_test, opt.label_test) ## get_eval_set函数中进行了归一化处理0-1
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')

model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')


def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')

    psnr_list = []
    ssim_list = []

    for batch in testing_data_loader:
        with torch.no_grad():
            input, label, name = batch[0], batch[1], batch[2]

            print(name)

        input = input.cuda()

        with torch.no_grad():
            j_out, t_out, tb_out = model(input)
            a_out = get_A(input).cuda()
            I_rec = j_out * t_out + (1 - tb_out) * a_out

            output_j = os.path.join(opt.output_folder, 'J') + os.sep
            output_a = os.path.join(opt.output_folder, 'A') + os.sep
            output_t = os.path.join(opt.output_folder, 'T') + os.sep
            output_r = os.path.join(opt.output_folder, 'R') + os.sep
            output_tb = os.path.join(opt.output_folder, 'TB') + os.sep
            for output_dir in [output_j, output_a, output_t, output_r, output_tb]:
                os.makedirs(output_dir, exist_ok=True)
            j_out_np = np.clip(torch_to_np(j_out), 0, 1)
            t_out_np = np.clip(torch_to_np(t_out), 0, 1)
            a_out_np = np.clip(torch_to_np(a_out), 0, 1)
            I_rec_np = np.clip(torch_to_np(I_rec), 0, 1)
            tb_out_np = np.clip(torch_to_np(tb_out), 0, 1)
            label_np = np.clip(torch_to_np(label), 0, 1)

            j_out_img = (j_out_np.transpose(1, 2, 0) * 255).round().astype(np.uint8)
            label_img = (label_np.transpose(1, 2, 0) * 255).round().astype(np.uint8)
            psnr = getPSNR(j_out_img, label_img)
            ssim_value = getSSIM(j_out_img, label_img)
            psnr_list.append(psnr)
            ssim_list.append(ssim_value)
            print("{} PSNR: {:.4f} | SSIM: {:.4f}".format(name[0], psnr, ssim_value))

            my_save_image(name[0], j_out_np, output_j)
            my_save_image(name[0], t_out_np, output_t)
            my_save_image(name[0], a_out_np, output_a)
            my_save_image(name[0], I_rec_np, output_r)
            my_save_image(name[0], tb_out_np, output_tb)

        ## 清理显存
        del input, label, j_out, t_out, tb_out, a_out, I_rec
        torch.cuda.empty_cache()

    if psnr_list:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        print('\nEvaluation metrics on test set:')
        print('Average PSNR: {:.4f}'.format(avg_psnr))
        print('Average SSIM: {:.4f}'.format(avg_ssim))
    else:
        print('\nNo test images were evaluated.')

if __name__ == '__main__':
    eval()



