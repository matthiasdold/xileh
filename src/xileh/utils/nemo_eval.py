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
        'nemo_log_dir': Path('/home/fr/fr_fr/fr_md1104/logs'),
        # for data transfer
        'nemo_shared_dir': Path('/home/fr/fr_fr/fr_md1104/tmp'),
        'local_pickle_tmp': Path('/tmp'),
        'singularity_container_home': '/work/ws/nemo/fr_md1104-singularity_container_0'      # noqa
    }
    return conf


def get_eval_sh():

    script = '''#!/bin/env bash
if [ -n "${SCRIPT_FLAGS}" ] ; then
    if [ -z "${*}" ]; then
        set -- ${SCRIPT_FLAGS}
    fi
fi

while [ "${1}" != "" ]; do
    case ${1} in
    -pl | --pipeline)		shift
                    PIPELINE_FILE=${1}
                                    ;;
        -d | --data)   			shift
                    DATA_FILE=${1}
                                    ;;
    -s | --singularity_path)	shift
                    SINGULARITY_PATH=${1};;
    * ) echo "illegal ARGUMENT ${1}"
    esac
    shift
done

echo "Loading singularity"
module load tools/singularity/3.5

# Note: make sure that the container is executeable
singularity exec $SINGULARITY_PATH/moabb_container_stups.simg python -c \
"import pickle; pl=pickle.load(open('$PIPELINE_FILE', 'rb')); data=pickle.load(open('$DATA_FILE', 'rb')); pl.eval(data)"
'''

    return script


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

    for dir in ['nemo_log_dir', 'nemo_shared_dir']:
        stdin, stdout, stderr = ssh_client.exec_command(f'mkdir {conf[dir]}')

        std_err_str = stderr.read()
        if std_err_str != b'' and b'File exists\n' not in std_err_str:
            NEMO_EVAL_LOGGER.error(b"Error on creating dirs: " + std_err_str)


def pack_pipelines_and_data_to_pickle(pdata):
    """ Create pickles to be transfered to nemo """
    NEMO_EVAL_LOGGER.info("Creating pickles for transport")

    conf = pdata.get_by_name('nemo_config').data

    # get unique pipelines and initial data -> transport only once
    pls = pdata.get_by_name('pipelines').data
    upls = list(set(pls))

    data = pdata.get_by_name('data').data
    udata = list(set(data))

    # Use the hash for identification
    pipeline_files = []
    data_files = []
    for pl in upls:
        fname = f'pipeline_{pl.__hash__()}.pickle'
        pickle.dump(pl, open(conf['local_pickle_tmp'].joinpath(fname), 'wb'))
        pipeline_files.append(fname)
    for ud in udata:
        fname = f'data_{ud.__hash__()}.pickle'
        pickle.dump(ud, open(conf['local_pickle_tmp'].joinpath(fname), 'wb'))
        data_files.append(fname)

    return {'pipeline_files': pipeline_files, 'data_files': data_files}


def transfer_local_to_nemo(ssh_client, conf, file_dict):
    NEMO_EVAL_LOGGER.info("Transfering files")
    ftp_client = ssh_client.open_sftp()
    local_tmp = conf['local_pickle_tmp']
    target_dir = conf['nemo_shared_dir']
    for files in file_dict.values():
        for fl in files:
            ftp_client.put(str(local_tmp.joinpath(fl)),
                           str(target_dir.joinpath(fl)))

    # also transfer the evaluation script -> msub on nemo will need a CMDFILE
    # i.e. shell script to execute --> this can be standardize and is
    # provided with xileh
    script_local_tmp_fl = conf['local_pickle_tmp'].joinpath(
        'eval_single_pipeline.sh')
    script_remote_fl = conf['nemo_shared_dir'].joinpath(
        script_local_tmp_fl.stem + script_local_tmp_fl.suffix)
    # script_local_tmp_fl.chmod(0x777)
    open(script_local_tmp_fl, 'w').write(get_eval_sh())
    ftp_client.put(str(script_local_tmp_fl),
                   str(script_remote_fl))

    ssh_client.exec_command(f'chmod 755 {script_remote_fl}')

    ftp_client.close()


def send_data_to_nemo(pdata):

    conf = pdata.get_by_name('nemo_config').data
    ssh_client = pdata.get_by_name('ssh_client').data
    prepare_for_transfer(conf, ssh_client=ssh_client)
    file_dict = pack_pipelines_and_data_to_pickle(pdata)

    transfer_local_to_nemo(ssh_client, conf, file_dict)


def initialize_ssh_connection_to_nemo(pdata, trg_container='ssh_client'):
    conf = pdata.get_by_name('nemo_config').data
    trg = pdata.get_by_name(trg_container, create_if_missing=True)

    trg.data = start_connection_to_nemo(conf)

    return pdata


def send_eval_jobs_to_nemo(pdata, jobs_container='nemo_jobs'):
    """ Create individual jobs for each pipeline/data pair from the containers
    """

    conf = pdata.get_by_name('nemo_config').data
    ssh_client = pdata.get_by_name('ssh_client').data

    # map data and files --> take the known directory and hashes
    pls = pdata.get_by_name('pipelines').data
    data = pdata.get_by_name('data').data

    pls_files = [(conf['nemo_shared_dir']
                  .joinpath(f'pipeline_{pl.__hash__()}.pickle'))
                 for pl in pls
                 ]

    data_files = [(conf['nemo_shared_dir']
                  .joinpath(f'data_{d.__hash__()}.pickle'))
                  for d in data
                  ]

    for pl_f, data_f in zip(pls_files, data_files):
        job_id = start_job(ssh_client, conf, pl_f, data_f)


def start_job(ssh_client, conf, pl_file, data_file):
    """ Start a job which will load a given pipeline and a given data
    container and will process pipeline.eval(data_container)

    Note: stdout and stderr are piped to a log file

    """
    log_file = conf['nemo_log_dir'].joinpath(
        pl_file.stem + '_' + data_file.stem + '.log')
    eval_script_path = str(conf['nemo_shared_dir'].joinpath(
        'eval_single_pipeline.sh'))

    # define the parameters to be passed to the script
    script_flags = f'-pl {pl_file} -d {data_file} '\
        f' -s {conf["singularity_container_home"]}'

    stdin, stdout, stderr = ssh_client.exec_command(
        f'msub -o {log_file} -e {log_file} -v SCRIPT_FLAGS="{script_flags}"'
        f' -l nodes=1:ppn=1,pmem=1gb,walltime=08:00:00 {eval_script_path}'
    )

    id_str = stdout.read().decode('ascii').replace('\n', '')

    # TODO: Check stderr

    if id_str == '':
        job_id = -1                     # something odd happened
    else:
        job_id = int(id_str)

    return job_id


def start_monitoring(pdata):
    """ Check the log files for the jobs by continously monitoring them
    over ssh via tail -f
    """
    pass


def test_print(pdata):
    print(pdata.get_containers())
    return pdata


# ==============================================================================
# Pipeline
# ==============================================================================
eval_pl = xPipeline('eval_on_nemo_pl')
eval_pl.add_step(('Check the input', validate_data, {}))
eval_pl.add_step(('Initialize ssh connection',
                  initialize_ssh_connection_to_nemo, {}))
eval_pl.add_step(('Send data to nemo',
                  send_data_to_nemo, {}))


if __name__ == '__main__':

    testpl = xPipeline('testpipeline')
    testpl.add_step(('testprint', test_print, {}))
    testpl2 = xPipeline('testpipeline')
    testpl2.add_step(('testprint', test_print, {}))
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

    eval_pl.eval(pdata)
