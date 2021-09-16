from numpy.lib.twodim_base import triu_indices
import pandas as pd
import tools_for_preprocessing as tpre
import tools_for_reconstruction as tr
import tools_for_postprocessing as tpost
import tools_for_visualization as tvis

## Choix du jeu de données :
dataset = "001"
# dataset = "002"
# dataset = "003"
# dataset = "004"
# dataset = "005"
# dataset = "006"
# dataset = "007"
# dataset = "008"
# dataset = "009"
# dataset = "010"

# Loading best parameters
best_parameters_from_file = pd.read_csv("best_parameters.csv", delimiter=" ")
best_parameters = dict(best_parameters_from_file.iloc[int(dataset)-1])

# Loading detections
detfile = "data_detection/" + dataset + "/detection.txt"
lens_param = tpre.get_lens_parameters(detfile)
raw_detections = pd.read_csv(detfile, delimiter=" ",skiprows=2)

# Preprocessing detections
clean_detections = tpre.get_clean_detections(dataset, best_parameters, min_frame = 0, max_frame = 1000, warm_start = False)

# Computing trajectories
trajectories = tr.get_trajectories_from_detections(clean_detections, best_parameters, allow_statio = True)

# Postprocessing trajectories
tpost.glue_trajectories(trajectories, best_parameters)
tpost.delete_small_trajectories(trajectories, best_parameters)

# Plotting trajectories
#on_image has to be False, for privacy concerns.
tvis.play_all_trajectories(trajectories, best_parameters, dataset, on_image=False, lens_param=lens_param, save = True, raw_detections = raw_detections)
