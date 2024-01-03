from skimage.metrics import structural_similarity
import numpy as np

def complex_psnr(x, y, peak = 'max'):
    '''

    :param x: reference image
    :param y: reconstructed image
    :param peak: normalized or max
    :return: psnr
    '''
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2 / mse)
    else:
        return 10 * np.log10(1. / mse)


def complex_nrmse(x, y, peak='max'):
    '''

    :param x: reference image
    :param y: reconstructed image
    :return: nrmse
    '''
    denom  = np.sqrt(np.mean((x*x),dtype=np.float64))
    mse = np.mean(np.abs(x - y) ** 2)
    out = np.sqrt(mse)/denom
    return out







def ssim(y_true, y_pred):
    ssim_sum=0
    y_true=y_true.cpu().detach().numpy()
    y_pred=y_pred.cpu().detach().numpy()
    for i in range(y_true.shape[0]):
        y_true = np.abs(y_true[i][0]+1j*y_true[i][1])
        y_pred = np.abs(y_pred[i][0]+1j*y_pred[i][1])
        score=structural_similarity(y_true, y_pred,data_range=y_true.max())
        ssim_sum=ssim_sum+score
    ssim_mean=ssim_sum/y_true.shape[0]
    return ssim_mean

def psnr(y_true, y_pred):
    psnr_sum=0
    y_true=y_true.cpu().numpy()
    y_pred=y_pred.cpu().detach().numpy()
    for i in range(y_true.shape[0]):
        y_true = y_true[i][0] + 1j * y_true[i][1]
        y_pred = y_pred[i][0] + 1j * y_pred[i][1]
        psnr=complex_psnr(y_true, y_pred)
        psnr_sum=psnr_sum+psnr
    psnr_mean=psnr_sum/y_true.shape[0]
    return psnr_mean

def nrmse(y_true, y_pred):
    y_true=y_true.cpu().numpy()
    y_pred=y_pred.cpu().detach().numpy()
    nrmse_sum=0
    for i in range(y_true.shape[0]):
        y_true = y_true[i][0] + 1j * y_true[i][1]
        y_pred = y_pred[i][0] + 1j * y_pred[i][1]
        nrmse = complex_nrmse(y_true, y_pred)
        nrmse_sum = nrmse_sum + nrmse
    nrmse_mean=nrmse_sum/y_true.shape[0]
    return nrmse_mean