from torch import optim
from utils.dataset import *
from utils.utils import *
from network.D2MC_Net import D2MC_Net
from utils.metric import *
from utils.loss import *
import argparse

if __name__ == '__main__':

    # parameters
    parser = argparse.ArgumentParser(description=' main ')
    parser.add_argument('--test_motion_path', default='/test/motion_data/', type=str,
                        help='motion data path for testing')
    parser.add_argument('--test_clean_path', default='/test/clean_data/', type=str,
                        help='clean data path for testing')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--outf', type=str, default='logs_D2MCNet', help='path of log files')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    args = parser.parse_args()

    device = torch.device('cuda', args.gpu_device if torch.cuda.is_available() else 'cpu')

    # load data info
    test_data_loader=load_data(args.batch_size,args.test_clean_path,args.test_motion_path)

    # Build model
    print('Loading model ...\n')
    net = D2MC_Net().to(device)
    net.load_state_dict(torch.load(os.path.join(args.outf, 'last_epoch_weights.pth')))
    net.eval()

    # test
    test_cor_ssim = 0.0
    test_cor_psnr = 0.0
    test_cor_nrmse = 0.0
    for iteration, batch in enumerate(test_data_loader):
        clean_image, motion_image, image_name = batch
        clean_image, motion_image = clean_image.to(device), motion_image.to(device)
        with torch.no_grad():
            rec_3, rec_2, rec_1, k_space_3, k_space_2, k_space_1 = net(motion_image)
            _cor_ssim = ssim(clean_image, rec_3).item()
            _cor_psnr = psnr(clean_image, rec_3)
            _cor_nrmse = nrmse(clean_image, rec_3)
            test_cor_ssim += _cor_ssim
            test_cor_psnr += _cor_psnr
            test_cor_nrmse += _cor_nrmse

    print('sim_mean_cor_ssim:', test_cor_ssim / len(test_data_loader))
    print('sim_mean_cor_psnr:', test_cor_psnr / len(test_data_loader))
    print('sim_mean_cor_nrmse:', test_cor_nrmse / len(test_data_loader))
