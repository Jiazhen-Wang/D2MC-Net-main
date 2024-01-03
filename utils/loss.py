from utils.metric import *

def ssim_loss(y_true, y_pred):
	loss_ssim = 1.0 - ssim(y_true, y_pred)
	return loss_ssim