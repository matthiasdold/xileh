import os
import pytest
import numpy as np

from xileh.core.pipeline import xPipeline
from xileh.core.pipelinedata import xData
from xileh.utils.logger import PlainLogger


def create_features(pdata, algo='C22'):
    meta = {f'catch22__{i}': 1 for i in range(1, 24)}
    pdata['TestData'].meta = meta

    return pdata


@pytest.fixture
def sample_data():
    tdata = xData(
        data=np.eye(5),
        header={'name': 'TestData',
                'description': 'Some data description'},
        meta={'mean': 5}
    )
    return tdata


@pytest.fixture
def sample_pipeline():
    sample_pipeline = xPipeline('testp', log_eval=False)
    return sample_pipeline


@pytest.fixture
def sample_pipeline_filled():

    sample_pipeline = xPipeline('testp', log_eval=False)
    sample_pipeline.add_steps(
        ('c22 extract', create_features, {'algo': 'c22'}),
        ('c22 extract 2', create_features),
        ('c22 extract 3', create_features, {})
    )
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


def test_set_step_kwargs(sample_pipeline_filled):
    # test overwriting existing and adding new ones
    kwargs = dict(algo='c42', more_kwargs_1=1, more_kwargs_2='a')

    sample_pipeline_filled.set_step_kwargs('c22 extract', **kwargs)

    # Note: Currently there is no check for whether a kwarg is valid for a func
    # consider added this

    step, idx = sample_pipeline_filled.get_step('c22 extract')

    for k, v in kwargs.items():
        assert step[2][k] == v


def test_eval_step_without_kwargs(sample_pipeline, sample_data):
    # a step added as (name, func) without kwargs must run fine
    sample_pipeline.add_step(('c22 extract', create_features))
    sample_pipeline.eval(sample_data)
    assert len([k for k in sample_data.meta.keys()
                if k.startswith('catch22__')]) == 23


def test_steps_setter_normalizes_kwargs(sample_pipeline, sample_data):
    # assigning raw 2-tuples via the setter must still be evaluable
    sample_pipeline.steps = [('c22 extract', create_features)]
    assert all(len(s) == 3 for s in sample_pipeline.steps)
    sample_pipeline.eval(sample_data)
    assert len([k for k in sample_data.meta.keys()
                if k.startswith('catch22__')]) == 23


def test_step_without_function_raises(sample_pipeline):
    # a (name, kwargs) step missing the function must fail early and clearly,
    # not as a cryptic "'dict' object is not callable" inside eval
    with pytest.raises(AssertionError):
        sample_pipeline.add_step(('mystep', {'algo': 'c22'}))


def test_step_with_non_dict_kwargs_raises(sample_pipeline):
    with pytest.raises(AssertionError):
        sample_pipeline.add_step(('mystep', create_features, 'not_a_dict'))


def test_len(sample_pipeline_filled):
    assert len(sample_pipeline_filled) == 3


def test_getitem_slice_returns_pipeline(sample_pipeline_filled):
    sub = sample_pipeline_filled[:2]
    assert isinstance(sub, xPipeline)
    assert len(sub) == 2
    assert sub.steps[0][0] == 'c22 extract'
    assert sub.steps[1][0] == 'c22 extract 2'


def test_getitem_int_returns_single_step_pipeline(sample_pipeline_filled):
    sub = sample_pipeline_filled[1]
    assert isinstance(sub, xPipeline)
    assert len(sub) == 1
    assert sub.steps[0][0] == 'c22 extract 2'


def test_getitem_negative_index(sample_pipeline_filled):
    sub = sample_pipeline_filled[-1]
    assert isinstance(sub, xPipeline)
    assert sub.steps[0][0] == 'c22 extract 3'


def test_getitem_preserves_settings():
    pl = xPipeline('mypl', verbose=True, silent=True, log_eval=True)
    pl.add_step(('s1', create_features))
    pl.add_step(('s2', create_features))
    sub = pl[:1]
    assert sub._name == 'mypl'
    assert sub.verbose is True
    assert sub._silent is True
    assert sub._log_eval is True


def test_getitem_slice_is_evaluable(sample_pipeline_filled, sample_data):
    sub = sample_pipeline_filled[:1]
    sub.eval(sample_data)  # must not raise


def test_steps_hash_initial_empty(sample_pipeline):
    assert sample_pipeline.steps_hash == sample_pipeline._compute_steps_hash()


def test_steps_hash_changes_on_add_step(sample_pipeline):
    h0 = sample_pipeline.steps_hash
    sample_pipeline.add_step(('s1', create_features))
    assert sample_pipeline.steps_hash != h0


def test_steps_hash_changes_on_remove_step(sample_pipeline):
    sample_pipeline.add_step(('s1', create_features))
    h1 = sample_pipeline.steps_hash
    sample_pipeline.remove_step('s1')
    assert sample_pipeline.steps_hash != h1


def test_steps_hash_changes_on_replace_step(sample_pipeline):
    sample_pipeline.add_step(('s1', create_features))
    h1 = sample_pipeline.steps_hash

    def other_fn(pdata):
        return pdata

    sample_pipeline.replace_step('s1', ('s1', other_fn))
    assert sample_pipeline.steps_hash != h1


def test_steps_hash_changes_on_set_step_kwargs(sample_pipeline):
    sample_pipeline.add_step(('s1', create_features, {'algo': 'A'}))
    h1 = sample_pipeline.steps_hash
    sample_pipeline.set_step_kwargs('s1', algo='B')
    assert sample_pipeline.steps_hash != h1


def test_steps_hash_changes_on_steps_setter(sample_pipeline):
    h0 = sample_pipeline.steps_hash
    sample_pipeline.steps = [('s1', create_features)]
    assert sample_pipeline.steps_hash != h0


@pytest.mark.parametrize('kwargs1,kwargs2,expect_equal', [
    ({'algo': 'A'}, {'algo': 'A'}, True),
    ({'algo': 'A'}, {'algo': 'B'}, False),
])
def test_steps_hash_same_steps_same_hash(kwargs1, kwargs2, expect_equal):
    pl1 = xPipeline('p1')
    pl2 = xPipeline('p2')
    pl1.add_step(('s1', create_features, kwargs1))
    pl2.add_step(('s1', create_features, kwargs2))
    assert (pl1.steps_hash == pl2.steps_hash) is expect_equal


def test_steps_hash_order_matters():
    pl1 = xPipeline('p1')
    pl2 = xPipeline('p2')
    pl1.add_steps(('s1', create_features), ('s2', create_features))
    pl2.add_steps(('s2', create_features), ('s1', create_features))
    assert pl1.steps_hash != pl2.steps_hash


def test_early_stop():
    """ An early stop would be signaled within the header of the xData """
    pdata = xData([], name='testing_data')
    pdata2 = xData([], name='testing_data')

    def add_data(pdata: xData, nbr: str = "1", add_stop: bool = False):
        pdata.add(nbr, name=nbr)

        if add_stop:
            pdata.header['early_stop'] = True

        return pdata

    pl = xPipeline("testpl")
    pl.add_steps(
        ("adddata1", add_data, {'nbr': "1"}),
        ("adddata2", add_data, {'nbr': "2", 'add_stop': True}),
        ("adddata3", add_data, {'nbr': "3"}),
    )
    pl2 = xPipeline("testpl", silent=True)
    pl2.add_steps(*pl.steps)
    pl.eval(pdata)
    pl2.eval(pdata2)

    assert pdata.get_container_names() == ["testing_data", "1", "2"]
    assert pdata2.get_container_names() == ["testing_data", "1", "2"]
    assert pdata.get_container_names() == ["testing_data", "1", "2"]

