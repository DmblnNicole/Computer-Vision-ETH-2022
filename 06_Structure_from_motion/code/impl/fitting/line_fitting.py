import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(0)
random.seed(0)

def least_square(x,y):
	# TODO
	# return the least-squares solution
	# you can use np.linalg.lstsq
	A = np.vstack([x, np.ones(len(x))]).T
	# y = kx+b
	k, b = np.linalg.lstsq(A, y, rcond=None)[0]
	return k, b

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# TODO
	# compute the number of inliers and a mask that denotes the indices of inliers
	# test for each data point (n_samples data points) if the distance is shorter than thresh_dist
	num = 0
	mask = np.zeros(x.shape, dtype=bool)
	for i in range(n_samples):
		# find shortest distance from each point to line y = kx+b
		dist = abs((k*x[i]+(-1)*y[i]+b)/np.sqrt((k*k)+((-1)*(-1))))
		if dist < thres_dist:
			# inlier
			num += 1
			mask[i] = True
	return num, mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	# TODO
	# ransac
	best_inliers_old = 0
	i = 0
	while i < iter:
		# sample from noisy data
		random_indices = random.sample(list(np.arange(n_samples)),num_subset)
		x_subset = x[random_indices]
		y_subset = y[random_indices]
		# compute best model parameters for subset
		k_ransac_old = least_square(x_subset, y_subset)[0]
		b_ransac_old = least_square(x_subset, y_subset)[1]
		# compute number of inliers for this parameters and the indices of the inliers
		inlier_mask_old = num_inlier(x, y, k_ransac_old, b_ransac_old, n_samples, thres_dist)[1]
		best_inliers_new = num_inlier(x, y, k_ransac_old, b_ransac_old, n_samples, thres_dist)[0]
		# if the number of inliers is larger than the current best result update k ransac, b ransac, inlier mask
		if best_inliers_new > best_inliers_old:
			k_ransac_new = k_ransac_old
			b_ransac_new = b_ransac_old
			inlier_mask_new = inlier_mask_old
		i += 1

	return k_ransac_new, b_ransac_new, inlier_mask_new

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()