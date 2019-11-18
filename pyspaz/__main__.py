
import click

@click.group()
def main():
    '''
    pyspaz is a module for analyzing live cell single molecule tracking data. The package includes localization, tracking, visualization, and analysis utilities.

    '''


@main.group()
def localize():
    '''
    Detection and localization of 2D Gaussian spots in an image
    with Poisson noise.
    '''
    pass

@localize.command()
@click.option('-i', '--input_file', type = str, help = 'Input file in .nd2 format (nikon) or TIF.')
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
    Detect and localize Gaussian spots in every frame of a single
    molecule tracking movie in either ND2 or TIF format.
    
    args
        input_file: str, a single ND2 or TIF file
        
        sigma: float, the expected standard deviation of the Gaussian spot

        out_txt: str, the name of a file to save the localizations, if desired

        out_dir: str, location to put output files. If None, these are placed
            in the same directory as the ND2 files
        
        window_size: int, the width of the window to use for spot detection
            and localization
        
        detect_threshold: float, the threshold in the log-likelihood image to 
            use when calling a spot
        
        damp: float, the factor by which to damp the update vector at each iteration
        
        camera_bg: float, the background level on the camera
        
        camera_gain: float, the grayvalues/photon conversion for the camera
        
        max_iter: int, the maximum number of iterations to execute before escaping
        
        plot: bool, show each step of the result for illustration
        
        initial_guess: str, the method to use for the initial guess. The currently
            implemented options are `radial_symmetry`, `centroid`, and `window_center`
        
        convergence: float, the criterion on the update vector for y and x when
            the algorithm can stop iterating
            
        divergence_crit: float, the criterion on the update vector for y and x
            when the algorithm should abandon MLE and default to a simpler method
            
        max_locs: int, the size of the localizations array to instantiate. This should
            be much greater than the number of expected localizations in the movie

        enforce_negative_definite : bool, whether to force the Hessian to be 
            negative definite by iteratively testing for negative definiteness
            by LU factorization, then subtracting successively larger ridge terms.
            If False, the method will only add ridge terms if numpy throws a
            linalg.linAlgError when trying to the invert the Hessian.

        verbose : bool, show the user the current progress 

    returns
        pandas.DataFrame with the localization results for this movie.
     
    '''
    from pyspaz.localize import detect_and_localize_file
    detect_and_localize_file(
        input_file,
        **kwargs
    )



@main.group()
def track():
    '''
    Track
    '''
    pass


if __name__ == '__main__':
    main()




