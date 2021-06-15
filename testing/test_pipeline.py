import pytest
import numpy as np

from xileh.core.pipeline import xPipeline
from xileh.core.pipelinedata import xPData
from xileh.core.features import create_features


@pytest.fixture
def sample_data():
    tdata = xPData(
        data=np.eye(5),
        header={'description': 'Some data description'},
        meta={'mean': 5}
    )
    return tdata


@pytest.fixture
def sample_pipeline():
    sample_pipeline = xPipeline('testp')
    return sample_pipeline


def test_add_step(sample_pipeline):
    sample_pipeline.add_step(('c22 extract', create_features, {'algo': 'c22'}))
    assert len(sample_pipeline._steps) == 1


def test_remove_step(sample_pipeline):
    sample_pipeline.add_step(('c22 extract', create_features, {'algo': 'c22'}))
    sample_pipeline.remove_step('c22 extract')
    assert len(sample_pipeline._steps) == 0


def test_ambiguous_step_name(sample_pipeline):
    # ambigous step names
    sample_pipeline.add_step(('c22 extract', create_features, {'algo': 'c22'}))
    with pytest.raises(AssertionError):
        sample_pipeline.add_step(
            ('c22 extract', create_features, {'algo': 'c22'}))


def test_simple_eval(sample_pipeline, sample_data):
    sample_pipeline.add_step(('c22 extract', create_features, {'algo': 'c22'}))
    sample_pipeline.eval(sample_data)
    assert len([k for k in sample_data.meta.keys()
                if k.startswith('catch22__')]) == 23
