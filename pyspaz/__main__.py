
import click

### order subcommand in help by their order in the code, not alphabetically 
class NaturalOrderGroup(click.Group):
    # def __init__(self, name=None, commands=None, **attrs):
    #     if commands is None:
    #         commands = OrderedDict()
    #     elif not isinstance(commands, OrderedDict):
    #         commands = OrderedDict(commands)
    #     click.Group.__init__(self, name=name,
    #                          commands=commands,
    #                          **attrs)

    def list_commands(self, ctx):
        return self.commands.keys()




@click.group(cls=NaturalOrderGroup)
def main():
    '''
    pyspaz is a module for analyzing live cell single molecule tracking data. The package includes localization, tracking, visualization, and analysis utilities.

    '''


@main.group(cls=NaturalOrderGroup)
def localize():
    '''
    Detection and localization of 2D Gaussian spots in an image with Poisson noise.
    '''
    pass

@localize.command()
@click.option('-i', '--input_file', type = str, required=True, help = 'Input file in .nd2 format (nikon) or TIF.')
@click.option('-ot', '--out_txt', type = str, default = None, help = ' default None')
@click.option('-od', '--out_dir', type = str, default = None, help = 'default None')
@click.option('-s', '--sigma', type = float, default = 1, help = 'the expected standard deviation of the Gaussian spot. default 1')
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9')
@click.option('-t', '--detect_threshold', type = float, default = 20.0, help = 'default 20.0')
@click.option('-d', '--damp', type = float, default = 0.2, help = 'default 0.2')
@click.option('-bg', '--camera_bg', type = float, default = 0.0, help = 'default 0.0')
@click.option('-gain', '--camera_gain', type = float, default = 110.0, help = 'default 110.0')
@click.option('-it', '--max_iter', type = int, default = 10, help = 'default 10')
@click.option('-g', '--initial_guess', type = str, default = 'radial_symmetry', help = 'default radial_symmetry')
@click.option('-cc', '--convergence_crit', type = float, default = 1.0e-3, help = 'default 1.0e-3')
@click.option('-dc', '--divergence_crit', type = float, default = 1.0, help = 'default 1.0')
@click.option('-m', '--max_locs', type = int, default = 2000000, help = 'default 2000000')
@click.option('-e', '--enforce_negative_definite/--no_enforce_negative_definite', default = False, help = 'default False')            
@click.option('-v', '--verbose',default = True, help = 'default True')            
def detect_and_localize_file(
    input_file, 
    **kwargs
):
    '''
    Detect and localize Gaussian spots in every frame of a single molecule tracking movie in either ND2 or TIF format.     
    '''
    from pyspaz.localize import detect_and_localize_file
    detect_and_localize_file(
        input_file,
        **kwargs
    )



@main.group(cls=NaturalOrderGroup)
def track():
    '''
    Tracking of single molecules from localisations.
    '''
    pass

@track.command()
@click.option('-i', '--input_loc_file', type = str, required=True, help = 'Input file of localisation in .csv format (output of pyspaz localize detect-and-localize-file).')
@click.option('-o', '--out_mat_file', type = str, default = None, help = 'Output .mat file.  default None')
@click.option('-dm', '--d_max', type = float, default = 20, help = 'default 20')
@click.option('-db', '--d_bound_naive', type = float, default = 0.1, help = 'default 0.1')
@click.option('-e', '--search_exp_fac', type = float, default = 3, help = 'default 3')
@click.option('-sep', '--sep', type = str, default = "\t", help = 'default \t')
@click.option('-p', '--pixel_size_um', type = float, default = 0.16, help = 'default 0.16')
@click.option('-f', '--frame_interval_sec', type = float, default = 0.00548, help = 'default 0.00548')
@click.option('-m', '--min_int', type = float, default = 0.0, help = 'default 0.0')
@click.option('-b', '--max_blinks', type = int, default = 0, help = 'default 0')
@click.option('-w', '--window_size', type = int, default = 9, help = 'default 9')
@click.option('-k', '--k_return_from_blink', type = int, default = 1, help = 'default 1')
@click.option('-yi', '--y_int', type = float, default = 0.5, help = 'default 0.5')
@click.option('-yd', '--y_diff', type = float, default = 0.9, help = 'default 0.9')
@click.option('--start_frame', type = int, default = None, help = 'default None')
@click.option('--stop_frame', type = int, default = None, help = 'default None')
@click.option('-v','--verbose', default = True, help = 'default True')
def track_locs(
    input_loc_file,
    **kwargs
):
    '''
    Track single molecules through frames from a localisation .csv file (from pyspaz localize detect-and-localize-file).
    '''
    from pyspaz.track import track_locs
    track_locs(
        input_loc_file,
        **kwargs
    )



@track.command()
@click.option('-i', '--input_spt_file', type = str, required=True, help = 'Input file of SPT in nd2 format.')
@click.option('-t', '--input_tracking_file', type = str, required=True, help = 'Input file of tracking in .mat format.')
@click.option('-s', '--start_frame', type = int, default=1000, help = 'default 1000')
@click.option('-e', '--end_frame', type = int, default=2000, help = 'default 2000')
@click.option('-o', '--out_tif', type = str, help = 'output tif file.')
@click.option('-v', '--vmax_mod', type = float, default = 1.0, help = 'default 1.0')
@click.option('-u', '--upsampling_factor', type = int, default = 1, help = 'default 1')
@click.option('-c', '--crosshair_len', type = int, default = 2, help = 'default 2')
@click.option('-p', '--pixel_size_um', type = int, default = 0.16, help = 'default 0.16')
@click.option('-p', '--plot_type', type=click.Choice(['localisation', 'currentTracks', 'allTracks']), default = 'localisation', help = 'default localisation')
def overlay_trajectories_on_spt(
    input_spt_file,
    input_tracking_file,
    start_frame,
    end_frame,
    **kwargs
):
    '''
    Create a movie Tif file with the original SPT data and its overlay with the localisation, colored by trajectory.
    '''
    from pyspaz.visualize import overlay_trajs
    overlay_trajs(
    input_spt_file,
    input_tracking_file,
    start_frame,
    end_frame,
    **kwargs
    )

@track.command()
@click.option('-t', '--input_tracking_file', type = str, required=True, help = 'Input file of tracking in .mat format.')
@click.option('-o', '--out_png', type = str, help = 'output tif file.')
@click.option('-c', '--cmap', type = str, default='viridis', help = 'Default viridis.')
@click.option('--cap', type = int, default=3000, help = 'default 3000')
@click.option('--n_colors', type = int, default=100, help = 'default 3000')
@click.option('--color_index', type = int, default=None, help = 'default None')
@click.option('-v','--verbose', default = True, help = 'default True')
def plot_tracked_mat(
    input_tracking_file,
    **kwargs
):
    '''
    xxx.
    '''
    from pyspaz.visualize import plot_tracked_mat
    plot_tracked_mat(
    input_tracking_file,
    **kwargs
    )


if __name__ == '__main__':
    main()




