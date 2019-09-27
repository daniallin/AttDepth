import math
import numpy as np
import numpy.ma as ma
import cv2
import time
import math

if __name__ == '__main__':
	dex = 0
	groundtruth_path = './dst/'
	calculated_path = './result/'
	sum_rmse = 0
	sum_mse = 0
	sum_L1=0
	sum_log_rmse=0

	count = 0
	avg_rmse = 0
	avg_mse = 0
	avg_L1=0
	avg_log_rmse=0

	while dex < 1201:
		start_t = time.time()
		dex_s = str(dex)
		dex_s = dex_s.zfill(6)
		print(dex_s)

		groundtruth_image = cv2.imread(groundtruth_path+dex_s+'.png',-1)
		print(groundtruth_image)
		calculated_image = cv2.imread(calculated_path+dex_s+'_pred.png',-1)
		print(calculated_image)
		if (groundtruth_image is not None) and (calculated_image is not None):
			if (groundtruth_image.shape[0] == calculated_image.shape[0]) and(groundtruth_image.shape[1]==calculated_image.shape[1]):
				groundtruth_image = groundtruth_image / 1000.0  # mm->m
				calculated_image = calculated_image /1000.0
				# rmse calculation#
				mask = (groundtruth_image > 10.0) | (groundtruth_image <= 0)
				mx = ma.array(groundtruth_image, mask=mask)
				mz = ma.array(calculated_image, mask=mask)

				tmp_rmse = np.sqrt(np.mean((mx - mz) ** 2))
				tmp_mse = np.mean((mx - mz) ** 2)
				tmp_L1 = np.mean(abs(mx - mz))
				tmp_log_rmse = np.sqrt(np.mean((np.log10(mx) - np.log10(mz)) ** 2))
				# rmse calculation#
				sum_rmse = avg_rmse * count + tmp_rmse
				sum_mse = avg_mse * count + tmp_mse
				sum_L1 = avg_L1 * count + tmp_L1
				sum_log_rmse = avg_log_rmse * count + tmp_log_rmse
				
				count = count + 1
				avg_rmse = sum_rmse / count
				avg_mse = sum_mse / count
				avg_L1 = sum_L1 / count
				avg_log_rmse = sum_log_rmse / count
				
				print('dex: ', dex, ' count: ', count, ' current_rmse: ', round(tmp_rmse, 3), ' avg_rmse: ',
				      round(avg_rmse, 3))
				print('dex: ', dex, ' count: ', count, ' current_mse: ', round(tmp_mse, 3), ' avg_mse: ',
				      round(avg_mse, 3))
				print('dex: ', dex, ' count: ', count, ' current_L1: ', round(tmp_L1, 3), ' avg_L1: ',
				      round(avg_L1, 3))
				print('dex: ', dex, ' count: ', count, ' current_log_rmse: ', round(tmp_log_rmse, 3), ' avg_log_rmse: ',
				      round(avg_log_rmse, 3))
				
				depthimage_colormap = cv2.applyColorMap(cv2.convertScaleAbs(calculated_image, alpha=30),
				                                        cv2.COLORMAP_JET)
				cv2.imshow('depth', depthimage_colormap)
				cv2.waitKey(1)
		dex = dex + 1
		#print('cost time: ',time.time()-start_t)
