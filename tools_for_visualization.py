import numpy as np
import matplotlib.pyplot as plt
import collections
import imageio
from scipy.signal import savgol_filter
import matplotlib.cm as cm

def get_frames(trajectory):
    return list(trajectory.keys())
def get_points(trajectory):
    return list(trajectory.values())

def plot_detections(detections, lens_param, frame, on_image):
    img_mask = (detections["#image"]==frame)
    x, y, z, h = detections[img_mask]["x"].values, detections[img_mask]["y"].values, detections[img_mask]["z"].values, detections[img_mask]["h"].values
    if on_image:
        f, cx, cy = lens_param
        ix = 0.5*(x/(z-h)*f+cx)
        iy = 0.5*(y/(z-h)*f+cy)
        plt.scatter(ix, iy, marker="+",color="grey");
    else:
        plt.scatter(x, y, marker="+",color="grey");

def to_image_coordinates(people, lens_param):
    f, cx, cy = lens_param
    ix = 0.5*(people[:,0]/people[:,2]*f+cx)
    iy = 0.5*(people[:,1]/people[:,2]*f+cy)
    ixmins, ixmaxs = np.zeros(ix.shape[0])+10, 448*np.ones(ix.shape[0])-10
    iymins, iymaxs = np.zeros(iy.shape[0])+10, 240*np.ones(iy.shape[0])-10
    return np.fmax(ixmins, np.fmin(ixmaxs, ix)), np.fmax(iymins, np.fmin(iymaxs, iy))

def _one_traj_on_image(trajectory, lens_param):
    points = np.array(get_points(trajectory))
    frames = get_frames(trajectory)
    points = np.transpose(list(to_image_coordinates(points, lens_param)))
    new_trajectory = dict()
    for k, frame in enumerate(frames):
        new_trajectory[frame] = points[k,:]
    return new_trajectory

def _smooth_one_traj(trajectory):
    if len(trajectory)>=5:
        points = np.array(get_points(trajectory))
        frames = get_frames(trajectory)
        wind = 5
        xs = savgol_filter(points[:,0], wind, 2)
        ys = savgol_filter(points[:,1], wind, 2)
        new_trajectory = dict()
        for k, frame in enumerate(frames):
            new_trajectory[frame] = np.array([xs[k], ys[k]])
        return new_trajectory
    else:
        return trajectory.copy()

def plot_trajectory_from_dict(trajectory, number, max_len=10, **kwargs):
    points = np.array(list(trajectory.values()))
    alphas = np.linspace(1, 1/max_len, max_len)
    kwargs['color'] = cm.get_cmap('gist_rainbow')(np.modf(number/7)[0])
    for i in range(min(points.shape[0]-1,max_len)):
        xs, ys = points[::-1, 0][i:i+2], points[::-1, 1][i:i+2]
        plt.plot(xs, ys, alpha = alphas[i], **kwargs)
    plt.scatter(points[-1,0], points[-1,1], marker='+', **kwargs)

def play_all_trajectories(trajectories, best_parameters, dataset, on_image=False, lens_param=None, save=False, raw_detections = None, smoothing=True):
    '''
    Plots trajectories computed from the data from dataset.

    Parameters
	----------
    trajectories : list
        List of trajectories (each of which of type dict).
    best_parameters : dict
		Dictionary containing parameters better suited to the data from dataset.
    dataset : str
        Name of the dataset which trajectories have been computed.
	on_image : bool, optional
		If True, plots trajectories on the original image.
	lens_param : tuple, optional
		Lens parameters of the camera for the data from dataset.
	save : bool, optional
		If True, saves the images of the plot in save/dataset/
	raw_detections : pandas.core.frame.DataFrame, optional
		Detections of the data from dataset without preprocessing. If None, it does not plot it.
    smoothing : bool, optional
        If True, smoothes the trajectories using _smooth_one_traj.
	'''

    #put every trajectory in order
    trajectories = [collections.OrderedDict(sorted(traj.items())) for traj in trajectories]
    #get the first and last frame
    min_frame_nb = np.min([ np.min(list(traj.keys())) for traj in trajectories ])
    max_frame_nb = np.max([ np.max(list(traj.keys())) for traj in trajectories ])

    domain = best_parameters['x_min'], best_parameters['x_max'], best_parameters['y_min'], best_parameters['y_max']

    if on_image:
        trajectories = [_one_traj_on_image(traj, lens_param) for traj in trajectories]

    if smoothing:
        trajectories = [_smooth_one_traj(traj) for traj in trajectories]

    plt.ion()
    fig = plt.figure(10)
    for frame in range(min_frame_nb, max_frame_nb+1):
        plt.clf()
        print('Frame: %s' %(frame))

        if on_image:
            imgdir  = "data_detection/" + dataset + "/images/"
            #Les images peuvent être au format png ou jpg
            try:
                im = imageio.imread(imgdir+"/image-"+str(frame).zfill(3)+".png")
            except:
                im = imageio.imread(imgdir+"/image-"+str(frame).zfill(3)+".jpg")
            plt.imshow(im)

        else:
            plt.axis(domain)

        if raw_detections is not None:
            plot_detections(raw_detections, lens_param,frame, on_image)

        title = 'frame : '+str(frame).zfill(3)
        for number, traj in enumerate(trajectories):
            traj_frames = get_frames(traj)
            first_fr, last_fr = np.min(traj_frames), np.max(traj_frames)

            if first_fr<= frame and frame <= last_fr:
                visible_traj = {k:traj[k] for k in range(first_fr, frame+1) if k in traj}
                plot_trajectory_from_dict(visible_traj, number)
        plt.title(title)
        plt.draw()
        if save:
            plt.pause(0.05)
            plt.savefig("save/"+dataset+"/image-"+str(frame).zfill(3)+".png")
        else:
            plt.pause(0.05)
    plt.clf()
    plt.ioff()
    plt.show()
