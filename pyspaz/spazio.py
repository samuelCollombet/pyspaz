'''
spazio.py -- I/O functions for tracking

'''
import pandas as pd
from scipy import io as sio

def trajs_to_mat(trajs, output_mat):
    '''
    Save trajectories to an output *Tracked.mat 
    file.

    args
    	trajs	:	2D ndarray of shape (n_trajs, m),
    				the trajectories and associated
    				information

    	output_mat	:	str, *Tracked.mat to save to

    '''
    out_dict = {'trackedPar' : take_trajs}
    sio.savemat(output_mat, out_dict)
    return out_dict

def save_trajectory_obj_to_mat(
    trajectories,
    mat_file_name,
    frame_interval_sec,
    pixel_size_um = 0.16,
    convert_pixel_to_um = True,
):
    '''
    Save a list of pyspaz.track.Trajectory objects to 
    *Tracked.mat file.

    args
        trajectories        :   list of pyspaz.track.Trajectory
        mat_file_name       :   str
        frame_interval_sec  :   float
        pixel_size_um       :   float
        convert_pixel_to_um :   bool

    returns
        dict, the read *Tracked.mat file

    '''
    result_list = []

    if convert_pixel_to_um:
        for traj_idx in range(len(trajectories)):
            trajectories[traj_idx].positions = trajectories[traj_idx].positions * pixel_size_um 

    for traj_idx, trajectory in enumerate(trajectories):
        result_list.append([
            trajectory.positions,
            trajectory.frames,
            [i * frame_interval_sec for i in trajectory.frames],
            trajectory.mle_I0,
            trajectory.mle_bg,
            trajectory.llr_detect,
        ])
    result = {'trackedPar' : result_list}
    
    sio.savemat(mat_file_name, result)
    return result 


def extract_positions_from_trajs(trajs):
    '''
    Return a 2D ndarray with all of the localization
    positions found in the trajectories.

    args
        trajs       :   trajectory object

    returns
        2D ndarray of shape (n_locs, 2), the y and x
            positions of each localization in um

    '''
    n_trajs = trajs.shape[0]
    n_locs = sum([trajs[traj_idx][0].shape[0] for traj_idx in range(N_trajs)])
    positions = np.zeros((n_locs, 2))
    loc_idx = 0
    for traj_idx in range(n_trajs):
        for _idx in range(trajs[traj_idx][0].shape[0]):
            positions[loc_idx, :] = trajs[traj_idx][0][_idx, :]
            loc_idx += 1
    return positions 

class ImageFileReader(object):
    '''
    Interface for grabbing frames from TIF or ND2 files.
    
        file_name: str, the name of a single TIF or ND2 file
        
    '''
    def __init__(
        self,
        file_name,
    ):
        self.file_name = file_name
        if '.nd2' in file_name:
            self.type = 'nd2'
            self.file_reader = ND2Reader(file_name)
            self.is_closed = False
        elif ('.tif' in file_name) or ('.tiff' in file_name):
            self.type = 'tif'
            self.file_reader = tifffile.TiffFile(file_name)
            self.is_closed = False 
        else:
            print('Image format %s not recognized' % \
                 os.path.splitext(file_name)[1])
            self.type = None
            self.is_closed = True 
        
    def get_shape(self):
        '''
        returns
            (int, int, int), the y dimension, x dimension, and
            t dimension of the data
        
        '''
        if self.is_closed:
            raise RuntimeError("Object is closed")
            
        if self.type == 'nd2':
            y_dim = self.file_reader.metadata['height']
            x_dim = self.file_reader.metadata['width']
            t_dim = self.file_reader.metadata['total_images_per_channel']
        elif self.type == 'tif':
            y_dim, x_dim = self.file_reader.pages[0].shape 
            t_dim = len(self.file_reader.pages)
        
        return (y_dim, x_dim, t_dim)
    
    def get_frame(self, frame_idx):
        '''
        args
            frame_idx: int
        
        returns
            2D np.array, the corresponding frame
        
        '''
        if self.is_closed:
            raise RuntimeError("Object is closed")
            
        if self.type == 'nd2':
            return self.file_reader.get_frame_2D(t = frame_idx)
        elif self.type == 'tif':
            return self.file_reader.pages[frame_idx].asarray()

    def close(self):
        self.file_reader.close()
        self.is_closed = True 




