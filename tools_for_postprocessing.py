import numpy as np
import tools_for_reconstruction as tr
import tools_for_visualization as vis

def get_trajectories(frame_start, frame_end, detections, domain): 
    images = np.unique(detections["#image"].values)
    current_trajectories, ended_trajectories, current_endpoints, current_labels = tr.init_trajectories(detections, images[0])

    for frame_start, frame_end in zip(images[:-1], images[1:]):
		
        people1, people2, trans_plan = tr.OT_step(frame_start, frame_end, detections, domain)
        var_for_traj = people1, people2, trans_plan
    
        tr.update_trajectories(current_trajectories, ended_trajectories, current_endpoints, current_labels, var_for_traj, frame_end)
    ended_trajectories += current_trajectories
    
    return ended_trajectories

def get_dead_trajectories(trajectories, dmax, domain): 
    """
    Extract from all trajectories the ones that have terminated at a distance >dmax from the domain
    """
    dead_traj=[]
    ind=[]
    for k, trajectory in enumerate(trajectories):
        last_point = vis.get_points(trajectory)[-1]
        if tr.dist_to_domain(last_point[:2], domain)>dmax**2:
            dead_traj.append(trajectory)
            ind.append(k)
    return dead_traj, ind
                    
def get_born_trajectories(trajectories, dmax, domain):
    """
    Extract from all trajectories the ones that have started at a distance greater than dmax from the domain
    """
    born_traj=[]
    ind=[]
    for k, trajectory in enumerate(trajectories):
        first_point = vis.get_points(trajectory)[0]
        if tr.dist_to_domain(first_point[:2], domain)>dmax**2:
            born_traj.append(trajectory)
            ind.append(k)
    return born_traj, ind  
    
def get_mappings_from_trajectories(trajectories, dmax, domain, depth):
    """
    Try to map a trajectory that has died too far from the domain with a one that has also started too far from the domain, at most depth frames afterwards.
    """
    mappings = []
    dead_traj, dead_ind = get_dead_trajectories(trajectories, dmax, domain)
    born_traj, born_ind = get_born_trajectories(trajectories, dmax, domain)
    for k in range(len(dead_traj)):
        traj_d = dead_traj[k]
        end_point = vis.get_points(traj_d)[-1]
        end_frames = [frame for frame,point in traj_d.items() if (point == end_point).all()]
        end_frames.sort()
        end_frame = end_frames[0]
        ll = born_traj.copy()
        for l, traj_b in enumerate(born_traj):
            first_frame = vis.get_frames(traj_b)[0]
            first_point = vis.get_points(traj_b)[0]
            if end_frame < first_frame <=end_frame + depth:
                if np.linalg.norm(end_point[:2]-first_point[:2])<(first_frame-end_frame)*dmax:
                    mappings.append((dead_ind[k], born_ind[l]))
                    ll.pop(l)
                    break
            born_traj = ll.copy()
    return mappings

def glue_trajectories_once(trajectories, best_parameters):
    """
    Perform the mapping and update the list of trajectories once
    """
    depth = best_parameters['max_statio']
    dmax = best_parameters['statio_thresh']
    domain = best_parameters['x_min'], best_parameters['x_max'], best_parameters['y_min'], best_parameters['y_max']
    end_indices =[]
    mappings = get_mappings_from_trajectories(trajectories, dmax, domain, depth)
    for (begin, end) in mappings:
        end_indices.append(end)
        trajectories[begin].update(trajectories[end])
    for index in sorted(end_indices, reverse = True):
        del trajectories[index]
        
def glue_trajectories(trajectories, best_parameters):
    """
    Update the list of trajectories where we have glued the ones that have stopped too far from the boundary of the domain with the ones that have started too far from the boundary of the domain.

    Parameters
    ----------
    trajectories : list
        List of trajectories (each of which of type dict).
    best_parameters : dict
		Dictionary containing parameters better suited to the data from dataset.
    """
    print('Gluing trajectories...', end = ' ')
    test = True
    while test:
        old_trajectories = trajectories.copy()
        glue_trajectories_once(trajectories, best_parameters)
        test = (old_trajectories != trajectories)
    print('Done.')

def delete_small_trajectories(trajectories, best_parameters):
    """
    Delete small trajectories from the list of trajectories, assumed to be unrealistic.
    
    Parameters
	----------
    min_size : int
        Minimal length of a trajectory that must be kept in the final list of trajectories.
    """
    print('Filtering small trajectories...', end = ' ')
    size = best_parameters['min_size']
    pop_ind =[]
    for k, trajectory in enumerate(trajectories):
        traj = vis.get_points(trajectory)
        if len(np.unique(traj, axis = 0))<=size:
            pop_ind.append(k)
    for index in sorted(pop_ind, reverse = True):
        del trajectories[index]
    print('Done.')