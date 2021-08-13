#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20210813
#
# Module for evaluating pipelines and data on nemo
#
# The general idea is to have a single function being able to
# -> pack up a pipeline and associated pipeline data into pickles
# -> tranfer it to nemo
# -> run the pipeline.eval(pipeline_data) as a job in a singularity container
# -> provide logging of the job and the pipelines
# -> (potentially) also retrieve the data

import pickle
import paramiko

from pathlib import Path

from xileh.core.pipelinedata import xPData
from xileh.core.pipeline import xPipeline
from xileh.utils.logger import DefaultLogger


NEMO_EVAL_LOGGER = DefaultLogger('nemo_eval_logger',
                                 log_file='/tmp/nemo_eval.log')


def get_default_config():
    conf = {
        'ssh_key_file': Path().home().joinpath('.ssh/id_rsa_nemo'),
        'nemo_host': 'login1.nemo.uni-freiburg.de',          # login node
        'nemo_user': 'fr_md1104',
        'log_dir': '/home/fr/fr_fr/fr_md1104/logs',
        'shared_dir': '/home/fr/fr_fr/fr_md1104/tmp',       # for data transfer
        'local_pickle_tmp': Path('/tmp')
    }
    return conf


def validate_data(pdata,
                  conf_container='nemo_config',
                  pl_container='pipelines',
                  data_container='data'):
    """
    Validate that all relevant data is present, i.e.
    nemo_conf, pipelines and data
    """
    NEMO_EVAL_LOGGER.info("Validating data container")

    # Containers are there
    for nm in [conf_container, pl_container, data_container]:
        assert nm in pdata.get_container_names(), "Missing container with "\
            f"name = '{nm}'"

    # Config is complete - at least keys for default config
    conf = pdata.get_by_name(conf_container).data
    missing_conf_keys = set(get_default_config().keys()) - set(conf.keys())
    assert missing_conf_keys == set(), "Config in container"\
        f" '{conf_container}' missing keys for '{missing_conf_keys}'"

    assert (len(pdata.get_by_name(pl_container).data)
            == len(pdata.get_by_name(data_container).data)), "Pipelines "\
        f" in '{pl_container}' and data in '{data_container}' do not match up"

    return pdata


def start_connection_to_nemo(conf):
    """ Create a paramiko SSH connection / client """
    NEMO_EVAL_LOGGER.info("Initilizing SSH connection")

    ssh_client = paramiko.SSHClient()

    # we add nemo to trusted hosts
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    k = paramiko.RSAKey.from_private_key_file(conf['ssh_key_file'])
    ssh_client.connect(hostname=conf['nemo_host'],
                       username=conf['nemo_user'], pkey=k)

    # stdin, stdout, stderr = ssh.exec_command('ls -l')
    # stdout.read()

    return ssh_client


def prepare_for_transfer(conf, ssh_client):
    """ Prepare directories at nemo """
    NEMO_EVAL_LOGGER.info("Preparing remote directories")

    for dir in ['log_dir', 'shared_dir']:
        stdin, stdout, stderr = ssh_client.exec_command(f'mkdir {conf[dir]}')

        std_err_str = stderr.read()
        if std_err_str != b'' and b'File exists\n' not in std_err_str:
            NEMO_EVAL_LOGGER.error(b"Error on creating dirs: " + std_err_str)


def pack_pipelines_and_data_to_pickle(pdata,
                                      pl_container='pipelines',
                                      data_container='data',
                                      ):
    """ Create pickles to be transfered to nemo """
    NEMO_EVAL_LOGGER.info("Creating pickles for transport")

    conf = pdata.get_by_name('nemo_config').data

    # get unique pipelines and initial data -> transport only once
    pls = pdata.get_by_name(pl_container).data
    upls = list(set(pls))
    data = pdata.get_by_name(data_container).data
    udata = list(set(data))

    # Use the hash for identification
    for pl in upls:
        fname = f'pipeline_{pl.__hash__()}.pickle'
        pickle.dump(pl, open(conf['local_pickle_tmp'].joinpath(fname), 'wb'))
    for ud in udata:
        fname = f'data_{ud.__hash__()}.pickle'
        pickle.dump(ud, open(conf['local_pickle_tmp'].joinpath(fname), 'wb'))


if __name__ == '__main__':

    testpl = xPipeline('testpipeline')
    testpl2 = xPipeline('testpipeline')
    testdt = xPData([1, 2, 3], name='testdata')
    testdt2 = xPData([1, 2, 3, 4], name='testdata')

    pdata = xPData(
        [
            xPData(get_default_config(), name='nemo_config'),
            xPData([testpl, testpl2, testpl], name='pipelines'),
            xPData([testdt, testdt2, testdt], name='data'),
        ],
        name='eval_on_nemo_data'
    )
