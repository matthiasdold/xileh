# use this to have a single source for the version, ie. pyproject.toml
import importlib.metadata
__version__ = importlib.metadata.version('xileh')


# get the main components for easier import. xPData is kept as a
# backwards-compatible alias for the renamed xData container.
from xileh.core.pipeline import xPipeline
from xileh.core.pipelinedata import xData, xPData
