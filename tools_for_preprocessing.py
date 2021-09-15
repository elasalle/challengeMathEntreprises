import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
import csv
import tools_for_preprocessing as tpre

def get_lens_parameters(detfile):
	file = open(detfile, "r")
	l0 = file.readline()
	l1 = file.readline()
	f,cx,cy = np.array(l1.split()).astype(int)
	file.close()
	return f,cx,cy

def remove_identical_points(x,y,z,h): 
	'''
	Removes duplicates of centroids having same (x,y)-coordinates.
	'''
	N_pts = len(x)
	coords = np.zeros((N_pts, 2))
	coords[:,0] = np.round(x,1)
	coords[:,1] = np.round(y,1)
	coords, indices = np.unique(coords, axis  = 0, return_index=True)
	
	x_uniq=coords[:,0]
	y_uniq=coords[:,1]
	z_uniq = z[indices]
	h_uniq = h[indices]
	return x_uniq, y_uniq, z_uniq, h_uniq

def get_centroids_from_detection(detection_of_img, domain, sigma, density_resolution = 100):
	'''
	Returns the centroids of the points detected on an image from its (preprocessed) detections.

	Parameters
	----------
	domain : 4-uple of xmin, xmax, ymin, ymax that determines the region where to compute the density f from detected points
	'''

	x = detection_of_img["x"].values
	y = detection_of_img["y"].values
	z = detection_of_img["z"].values
	h = detection_of_img["h"].values
	nbr_pts = x.shape[0] # number of detected points

	xmin, xmax, ymin, ymax = domain

	if nbr_pts != 0:
		# Constructing density function f as sum of gaussians centered 
		X = np.linspace(xmin, xmax, density_resolution)
		Y = np.linspace(ymin, ymax, density_resolution)
		xx,yy = np.meshgrid(X,Y)
		f = np.zeros(xx.shape)
		for j in range(nbr_pts):
			f += h[j]*np.exp(-(xx-x[j])**2/(2*sigma**2) - (yy-y[j])**2/(2*sigma**2))

		# Computing maxima of density function
		maxima = peak_local_max(f, min_distance = 2, exclude_border=False)
		# print('maxima: \n %s' %maxima)
		x_maxima = X[maxima[:,1]]
		y_maxima = Y[maxima[:,0]]

		# Computing heights and depth of maxima
		maxima_as_pts = np.transpose([x_maxima,y_maxima])
		detected_pts = np.transpose([x,y])
		distances_maxima_to_pts = cdist(maxima_as_pts, detected_pts)
		ind_closest = list(np.argmin(distances_maxima_to_pts, axis = -1))

		z_maxima = z[ind_closest]
		h_maxima = h[ind_closest]
	else:
		x_maxima, y_maxima, z_maxima, h_maxima = np.array([]), np.array([]), np.array([]), np.array([])

	# return x_maxima, y_maxima, z_maxima, h_maxima
	return remove_identical_points(x_maxima, y_maxima, z_maxima, h_maxima)

def get_clean_detections(dataset, best_parameters, warm_start = False, min_frame = 0, max_frame = 1000):
	'''
	Returns and saves fully preprocessed detections from raw detections following the pipeline:
		1. Deleting outliers in detections
			- Height threshold by height_threshold in best_parameters
			- Frames selection (optional) between min_frame and max_frame
			- Geometric constraints: outside the rectangle [x_min, x_max]x[y_min,y_max] is considered as noise). The parameters x_min, x_max, y_min, y_max are taken from best_parameters.
		2. Replacing detected points by the centroids computed with get_centroids_from_detection.

	Parameters
	----------
	dataset : str
		Name of the dataset whose detections are preprocessed.
	best_parameters : dict
		Dictionary containing parameters better suited to the data from dataset.
	warm_start : bool, optional
		If True, loads clean_detections corresponding to the data from dataset without computing it.
	min_frame : int, optional
		Index of frame at which clean_detections will start.
	max_frame : int, optional
		Index of frame at which clean_detections will end.

	Returns
	-------
	clean_detections : pandas.core.frame.DataFrame
		Preprocessed detections.
	'''
	if not warm_start:
		print('Cleaning detections...')
		detfile = "data_detection/" + dataset + "/detection.txt"
		f, cx, cy = tpre.get_lens_parameters(detfile)
		detections = pd.read_csv(detfile, delimiter=" ",skiprows=2)

		sigma = best_parameters['sigma']
		domain = best_parameters['x_min'], best_parameters['x_max'], best_parameters['y_min'], best_parameters['y_max']
		
		# Preprocessing data 
		height_mask = detections["h"]>best_parameters['height_thresh']
		minmax_frame_mask = (detections["#image"]>=min_frame) & (detections["#image"]<max_frame)
		geometric_mask = (detections["x"]>best_parameters['x_min']) & (detections["x"]<best_parameters['x_max']) & (detections["y"]>best_parameters['y_min']) & (detections["y"]<best_parameters['y_max'])
		preprocessing_mask = height_mask & minmax_frame_mask & geometric_mask
		preprocessed_detections = detections[preprocessing_mask]

		# Computing coordinates of centroids
		images = np.unique(preprocessed_detections["#image"].values)
		
		clean_data = open('csv/'+ dataset + '_clean_data.csv', 'w') #Enregistre les nouvelles données en csv
		clean_data.write("#image x y z h\n")
		clean_data.close()
		clean_data = open('csv/'+ dataset + '_clean_data.csv', 'a') #Enregistre les nouvelles données en csv
		writer = csv.writer(clean_data, delimiter = " ")
		for i in images:
			# print('image: %s' %i)
			img_mask = (preprocessed_detections["#image"]==i)
			x, y, z, h = get_centroids_from_detection(preprocessed_detections[img_mask], domain, sigma)
			nbr_pts = x.shape[0]
			for k in range(nbr_pts):
				writer.writerow([i, x[k], y[k], z[k], h[k]])
		clean_data.close()
		print('Detections cleaned.')

	clean_detections = pd.read_csv('csv/%s_clean_data.csv' %(dataset), delimiter = " ")
	return clean_detections