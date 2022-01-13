import os
import pytest
import numpy as np

from xileh.core.pipeline import xPipeline
from xileh.core.pipelinedata import xPData
from xileh.core.features import create_features
from xileh.utils.logger import PlainLogger


@pytest.fixture
def sample_data():
    tdata = xPData(
        data=np.eye(5),
        header={'name': 'TestData',
                'description': 'Some data description'},
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


def test_add_steps(sample_pipeline):
    sample_pipeline.add_steps(
        ('c22 extract', create_features, {'algo': 'c22'}),
        ('c22 extract 2', create_features),
        ('c22 extract 3', create_features, {})
    )
    assert len(sample_pipeline._steps) == 3


def test_replace_step(sample_pipeline):
    sample_pipeline.add_steps(
        ('c22 extract', create_features, {'algo': 'c22'}),
        ('c22 extract 2', create_features),
        ('c22 extract 3', create_features, {})
    )

    sample_pipeline.replace_step('c22 extract 2',
                                 ('new_step', create_features))

    assert sample_pipeline._steps[1][0] == 'new_step'


def test_get_step(sample_pipeline):
    sample_pipeline.add_step(('c22 extract', create_features, {'algo': 'c22'}))
    assert (sample_pipeline.get_step('c22 extract')[0]
            == sample_pipeline._steps[0])

    # add some more and check the index
    sample_pipeline.add_step(
        ('c22 extract2', create_features, {'algo': 'c22'}))
    assert (sample_pipeline.get_step('c22 extract2')[1]
            == 1)


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


def test_logging(sample_pipeline, sample_data, tmpdir):
    sample_pipeline.add_step(('c22_1', create_features, {'algo': 'c22'}))
    sample_pipeline.add_step(('c22_2', create_features, {'algo': 'c22'}))

    sample_pipeline._log_eval = True
    logfile = tmpdir.join(f'{sample_pipeline._name}.log')

    sample_pipeline._logger = PlainLogger(logfile)
    sample_pipeline.eval(sample_data)

    assert os.path.exists(logfile), f"No log file at {logfile}"

    lines = open(logfile, 'r').readlines()
    assert len(lines) == 5
    assert "Finished step 2/2: c22_2" in lines[-1]
