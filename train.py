from torch import optim
from utils.dataset import *
from utils.utils import *
from network.D2MC_Net import D2MC_Net
from utils.metric import *
from utils.loss import *
import argparse
from torch.nn import functional as F


if __name__ == '__main__':

    # parameters
    parser = argparse.ArgumentParser(description=' main ')
    parser.add_argument('--train_motion_path', default='/train/motion_data/', type=str,
                        help='motion data path for training')
    parser.add_argument('--train_clean_path', default='/train/clean_data/', type=str,
                        help='clean data path for training')
    parser.add_argument('--test_motion_path', default='/test/motion_data/', type=str,
                        help='motion data path for testing')
    parser.add_argument('--test_clean_path', default='/test/clean_data/', type=str,
                        help='clean data path for testing')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--outf', type=str, default='logs_D2MCNet', help='path of log files')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    args = parser.parse_args()

    device = torch.device('cuda', args.gpu_device if torch.cuda.is_available() else 'cpu')

    try_make_dir(args.outf)


    # callable methods
    def adjust_learning_rate(opt):
        for param_group in opt.param_groups:
            param_group['lr'] *= 0.1


    # dataset
    train_data_loader=load_data(args.batch_size,args.train_clean_path,args.train_motion_path)
    test_data_loader=load_data(args.batch_size,args.test_clean_path,args.test_motion_path)

    # D2MC_Net model
    net = D2MC_Net().to(device)

    # Adam optimizer
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-5)

    # train
    print('========== start train ================')
    net.train()
    for epoch in range(args.num_epoch):
        run_loss = 0.0
        run_psnr = 0.0

        if epoch == 40:
            adjust_learning_rate(opt)


        for iteration, batch in enumerate(train_data_loader):
            clean_image, motion_image,_ = batch
            clean_image, motion_image = clean_image.to(device), motion_image.to(device)
            clean_kspace = fft2(clean_image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            rec_3, rec_2, rec_1, k_space_3, k_space_2, k_space_1 = net(motion_image)

            opt.zero_grad()

            total_loss = F.l1_loss(clean_image, rec_1) + ssim_loss(clean_image, rec_1) + \
                         F.l1_loss(clean_image, rec_2) + ssim_loss(clean_image, rec_2) + \
                         F.l1_loss(clean_image, rec_3) + ssim_loss(clean_image, rec_3) + \
                         0.001 * ( F.l1_loss(clean_kspace, k_space_1) + F.l1_loss(clean_kspace, k_space_2)+F.l1_loss(
                clean_kspace, k_space_3))
            total_loss.backward()
            opt.step()
            run_loss += total_loss.item()
            cor_psnr = psnr(clean_image, rec_3)
            run_psnr += cor_psnr
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, iteration + 1, len(train_data_loader), run_loss / (iteration + 1),
                   run_psnr / (iteration + 1)))

        run_loss /= len(train_data_loader)
        run_psnr /= len(train_data_loader)
        print("train_loss: ", run_loss)
        print("train_psnr: ", run_psnr)

        # save model
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(
                args.outf,'model{}.pth'.format(epoch+1)))
        torch.save(net.state_dict(), os.path.join(args.outf, "last_epoch_weights.pth"))

    # test
    print('========== start test ================')
    net.eval()
    test_cor_ssim = 0.0
    test_cor_psnr = 0.0
    test_cor_nrmse = 0.0

    for iteration, batch in enumerate(test_data_loader):
        clean_image, motion_image,image_name = batch
        clean_image, motion_image = clean_image.to(device), motion_image.to(device)
        with torch.no_grad():
            rec_3, rec_2, rec_1, k_space_3, k_space_2, k_space_1 = net(motion_image)
            _cor_ssim = ssim(clean_image,rec_3).item()
            _cor_psnr = psnr(clean_image, rec_3)
            _cor_nrmse = nrmse(clean_image, rec_3)
            test_cor_ssim += _cor_ssim
            test_cor_psnr += _cor_psnr
            test_cor_nrmse += _cor_nrmse

    print('sim_mean_cor_ssim:', test_cor_ssim /len(test_data_loader))
    print('sim_mean_cor_psnr:', test_cor_psnr / len(test_data_loader))
    print('sim_mean_cor_nrmse:', test_cor_nrmse / len(test_data_loader))

