import numpy as np
import ot

def dist_to_domain(point, domain):
    y = point[1]
    ymin, ymax = domain[2], domain[3]
    d1 = (y-ymin)**2
    d2 = (y-ymax)**2
    return np.min([d1, d2])

def dists_to_domain(x, domain):
    n = len(x)
    dists = np.array([ dist_to_domain(x[i,:], domain) for i in range(n)])
    return dists

def compute_data_distr_with_border(n1, n2):
    """
    Assume all points have equal mass (except the domain) in order to perform optimal transport
    """   
    a = np.full(n1+1, 1. / (n1 + n2) )
    b = np.full(n2+1, 1. / (n1 + n2) )
    a[-1] = a[-1] * n2
    b[-1] = b[-1] * n1
    return a,b

def complete_dist_with_dist_to_domain(M, dist1, dist2):
    """
    Complete the matrix of distance M with the distance of each point to the domain boundary.
    """
    n1, n2 = M.shape
    dist1 = dist1.reshape((n1,1))
    dist2 = dist2.reshape((1,n2))
    dist2 = np.concatenate([dist2, np.array([[0]])], axis=1)
    M = np.concatenate((M, dist1), axis=1)
    M = np.concatenate((M, dist2), axis=0)
    return M

def get_coordinate_pts_from_detections(detections, frame):
    mask = (detections["#image"]==frame)
    x = detections[mask]["x"].values
    y = detections[mask]["y"].values
    z = detections[mask]["z"].values
    h = detections[mask]["h"].values
    n1 = len(x)
    people = np.zeros((n1, 3))
    people[:,0] = x
    people[:,1] = y
    people[:,2] = z-h
    return people

def _penalization(sqdist, statio_thresh=50, statio_time=None, slope=2):
    if statio_time is None:
        times = np.ones(sqdist.shape[0])
    else:
        times = np.array(statio_time+[1]) #add a statio time of 1 for the border
    T = statio_thresh**2
    threshs =  times.reshape((-1,1))**2 * T*np.ones(sqdist.shape)
    index = np.where(sqdist>threshs)
    sqdist[index] = slope*(sqdist[index] - threshs[index]) + threshs[index]
    return sqdist

def OT_step(frame_start, frame_end, detections, domain, pen=1, allow_statio=False, threshold=50, people1=None, statio_time=None, max_statio=10, statio_slope=2):  
    """
     Perform optimal transport between two frames

    Parameters
    ----------
    frame_start :TYPE int
    frame_end : TYPE int
    detections : TYPE pandas.core.frame.DataFrame
		DESCRIPTION Detections of a sequence of frames from which trajectories should be recovered.
    domain : TYPE tuple
        Rectangle on which to project
    pen : TYPE float
        DESCRIPTION. The default is 1. Amount of penalty in height
    allow_statio : TYPE Boolean
        DESCRIPTION. The default is False. Wether we consider variation of optimal transport where we allow
        a point to persist during multiple frames if there is a missing datum
    threshold : TYPE float
        DESCRIPTION. The default is 50. Maximal distance at which the displacement is heavily penalized
    people1 : TYPE array
    DESCRIPTION People from the first frame
    statio_time : TYPE list
        DESCRIPTION : number of frames each person has stationated.
    max_statio : TYPE int
        DESCRIPTION. The default is 10. Longest time a trajectory is allowed to stagnate
    statio_slope : TYPE float
        DESCRIPTION. Slope controlling the amount of penalization for a displacement longer than dmax

    Returns
    -------
    people1 : TYPE
        DESCRIPTION.
    people2 : TYPE
        DESCRIPTION.
    trans_plan : TYPE
        DESCRIPTION.

    """
   
    
    if allow_statio:
        forbiden_value    =  10**8  #for forbiden matchings
        
        if people1 is None:
            people1 = get_coordinate_pts_from_detections(detections, frame_start)
        elif type(people1)==list:
            people1 = np.array(people1)
        else:
            TypeError("people1 must be None or list, but received {}".format(type(people1)))
        n1 = people1.shape[0]
        
        people2 = get_coordinate_pts_from_detections(detections, frame_end)
        n2 = people2.shape[0]
        people2 = np.concatenate((people2, people1), axis=0)
        ntot = n1+n2
        
    else:
        people1 = get_coordinate_pts_from_detections(detections, frame_start)
        people2 = get_coordinate_pts_from_detections(detections, frame_end)
        n1, n2 = people1.shape[0], people2.shape[0]
        ntot = n2
        
    #data distribution
    distr1, distr2 = compute_data_distr_with_border(n1, ntot)
        
    # loss matrix
    M = ot.dist(people1[:,:2], people2[:n2,:2]) +pen*ot.dist(people1[:,2].reshape(-1, 1), people2[:n2,2].reshape(-1, 1))  #square distances
    dist1 =  dists_to_domain(people1[:,:2], domain)
    dist2 =  dists_to_domain(people2[:n2,:2], domain)
    M = complete_dist_with_dist_to_domain(M, dist1, dist2)
        
    if allow_statio:
        M = _penalization(M, threshold, statio_time, statio_slope)
        Mbis = np.ones((n1,n1))*forbiden_value
        diagonal = (threshold * np.array(statio_time))**2
        index = np.where(np.array(statio_time)>=max_statio)
        diagonal[index] = forbiden_value
        np.fill_diagonal(Mbis, diagonal)
        Mbis = np.concatenate((Mbis, np.zeros((1,n1))), axis=0)
        M[index,-1] = 0
        M = np.concatenate((M[:,:-1], Mbis, M[:,-1].reshape((-1,1))), axis=1)
        
    trans_plan = ot.emd(distr1, distr2, M)
    return people1, people2, trans_plan

def init_trajectories(detections, frame):
    current_trajectories, ended_trajectories, current_endpoints = [], [], []
    people = get_coordinate_pts_from_detections(detections, frame)
    for i in range(len(people)):
        current_trajectories.append({frame:people[i]})
        current_endpoints.append(people[i])
    statio_time = [1]*len(people)
    return current_trajectories, ended_trajectories, current_endpoints, statio_time

def update_trajectories(current_trajectories, ended_trajectories, current_endpoints,var_for_trajectories, frame, allow_statio=False, statio_time=None):
    ppl1, ppl2, trans_plan = var_for_trajectories
    n1, ntot = ppl1.shape[0], ppl2.shape[0]
    if allow_statio:
        n2 = ntot-n1
    else:
        n2 = ntot
    ind_deaths = np.where(trans_plan[:-1,-1]>1e-10)[0]
    ind_births = np.where(trans_plan[-1,:-1]>1e-10)[0]
    for i in ind_deaths: # Ending corresponding trajectories
        ind_traj = np.where(np.all(current_endpoints == ppl1[i],1))[0][0] # corresponding trajectory
        current_endpoints.pop(ind_traj)
        ended_traj = current_trajectories.pop(ind_traj)
        ended_trajectories.append(ended_traj)
        if allow_statio:
            statio_time.pop(ind_traj)
    for i in ind_births: # adding new corresponding trajectories
        if i<n2:
              current_trajectories.append({frame: ppl2[i]})
              current_endpoints.append(ppl2[i])
              if allow_statio:
                  statio_time.append(1)
    for i in range(n1): # updating corresponding trajectories
        ind_match = np.where(trans_plan[i,:-1]>1e-10)[0]
        if ind_match.shape[0]!= 0:
            ind_traj = np.where(np.all(current_endpoints == ppl1[i],1))[0][0] # corresponding trajectory
            target_point = ppl2[ind_match[0]]
            current_trajectories[ind_traj][frame] = target_point
            current_endpoints[ind_traj] = target_point
            if allow_statio:
                if ind_match>=n2:
                    statio_time[ind_traj] += 1  #the trajectory is stationary, its statio time is incresed by 1
                else:
                    statio_time[ind_traj]  = 1  #the trajectory is moving, its statio time is set to 1 again

def get_trajectories_from_detections(detections, best_parameters, allow_statio = False):
    '''
    Reconstruct trajectories of people from the (preprocessed) detections of a sequence of frames.

    Parameters
	----------
	detections : pandas.core.frame.DataFrame
		Detections of a sequence of frames from which trajectories should be recovered.
	best_parameters : dict
		Dictionary containing parameters better suited to the data from which the detections come from.
	allow_statio : bool, optional
		If True, allows stationarity in the reconstruction of trajectories using OT_step.

	Returns
	-------
	trajectories : list
        List of trajectories, each of which of type dict with keys the indexes of frames (type int) and values the corresponding points of the trajectory at the frame (type numpy.ndarray of size (3,)).
	'''
    print('Getting trajectories...', end = ' ')
    images = np.unique(detections["#image"].values)
    first_frame = detections['#image'].iloc[0]
    current_trajectories, ended_trajectories, current_endpoints, statio_time = init_trajectories(detections, first_frame)
    domain = best_parameters['x_min'], best_parameters['x_max'], best_parameters['y_min'], best_parameters['y_max']
    pen = best_parameters['pen']
    statio_thresh = best_parameters['statio_thresh']
    max_statio = best_parameters['max_statio']
    statio_slope = best_parameters['statio_slope']

    # Updating trajectories
    for current_frame, next_frame in zip(images[:-1], images[1:]):
        people1, people2, trans_plan = OT_step(current_frame, next_frame, detections, domain, pen, allow_statio, statio_thresh, current_endpoints, statio_time, max_statio, statio_slope)
        var_for_traj = people1, people2, trans_plan
        update_trajectories(current_trajectories, ended_trajectories, current_endpoints, var_for_traj, next_frame, allow_statio, statio_time)
    ended_trajectories += current_trajectories
    print('Done.')
    return ended_trajectories