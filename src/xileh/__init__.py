# use this to have a single source for the version
# ie. the setup.cfg
import importlib.metadata
__version__ = importlib.metadata.version('xileh')


# get the two main components for easier import
from xileh.core.pipeline import xPipeline
from xileh.core.pipelinedata import xPData
