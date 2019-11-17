'''
__init__.py
'''
import pyspaz.spazio
import pyspaz.utils
import pyspaz.visualize
import pyspaz.localize
import pyspaz.track
import pyspaz.mask

# Command line interface - this is currently a skeleton
# import click

# @click.group()
# def cli():
#     pass

# @cli.command()
# @click.argument('nd2_file', type = str)
# @click.option('-n', '--n_files', type = int, default = 0)
# def detect_and_localize(
#     nd2_file,
#     **kwargs,
# ):
#     print(nd2_file)
#     print(kwargs['n_files'])

#     # Something like this
#     # pyspaz.localize.localize_file(
#     #     *args,
#     #     **kwargs,
#     # )

# @cli.command()
# @click.argument('tracked_mat_file', type = str)
# def track(tracked_mat_file):
#     print(tracked_mat_file)

# cli()