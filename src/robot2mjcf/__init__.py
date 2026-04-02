__version__ = "0.1.23"

from .cli.convert import convert_urdf_to_mjcf as run

__all__ = ["run", "__version__"]
