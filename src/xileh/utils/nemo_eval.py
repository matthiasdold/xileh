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
#
# ---- Some notes: --------------------------------------------------------
#
# Doing the evaluation at nemo for pipelines and data pairs will create
# two type of log files at the nemo_log_dir:
#   - a log file <pipeline_name>.log => will track the pipeline logging
#     via log_eval=True and is necessary for the monitoring, as this will
#     include the information about which step is completed etc
#     ==> hence it is required that the pipelines have unique names!
#   - a log file pipeline_<pipeline_hash>_data_<data_hash>.log which
#     includes all the STDOUT and STDERR produced by evaluating the pipeline
#     e.g. a simply print("something") will write a line with "something"
#     this is also used to check if an error in evaluting the data on the
#     pipeline occured
#
# Some usefull nemo commands:
#   `qstat -f` - shows detailed info on all submitted jobs
#
#
#

import os
import dill
import paramiko
import inspect

from pathlib import Path
from time import sleep
from subprocess import Popen

from xileh.core.pipelinedata import xPData
from xileh.core.pipeline import xPipeline
from xileh.utils.monitoring import PipelineMonitor


# Note: The plain logger can be pickled and sent to nemo
from xileh.utils.logger import DefaultLogger, PlainLogger


NEMO_EVAL_LOGGER = DefaultLogger('nemo_eval_logger',
                                 log_file='/tmp/nemo_eval.log')


def get_default_config():
    conf = {
        'ssh_key_file': Path().home().joinpath('.ssh/id_rsa_nemo'),
        'nemo_host': 'login1.nemo.uni-freiburg.de',          # login node
        'nemo_user': 'fr_md1104',
        'nemo_log_dir': Path('/home/fr/fr_fr/fr_md1104/logs'),
        'local_log_dir': Path('/tmp/nemo_eval/'),
        # for data transfer
        'nemo_shared_dir': Path('/home/fr/fr_fr/fr_md1104/tmp'),
        'local_pickle_tmp': Path('/tmp'),
        'data_root_nemo': Path('/work/ws/nemo/'),
        'data_root_local': Path('/work/ws/nemo/'),                                  # we replicate the local root with the bind mount command upon singularity start -> no need to adjust the paths in the scripts
        'singularity_container_dir': Path('/work/ws/nemo/fr_md1104-singularity_container-0/'),      # noqa
        'singularity_container_name': 'xileh_pd_interactive.sif',      # noqa
        # For unpickling to be possible, we need the scripts, including __main__
        # Everything from the parent directory of __main__ downwards will be
        # copied to local_log_dir on the remote host
        # TODO: Consider how this could be derived dynamically
        '__main__': Path('/home/doda/workspace/python/xileh/src/xileh/utils/nemo_eval.py'),                      # noqa
        # Take all the script roots from this list and copy all *.py
        'script_paths_to_copy': [],
        'ram_to_use': '1500mb',
        'time_to_allocate': '00:10:00',
        'queue_to_use': '',
    }
    return conf


def get_eval_sh():

    script = '''#!/bin/env bash
                # Note: for testing this script you can start an interacitve session and run
                # it against some copied pickles
                #
                # msub  -I  -V  -l nodes=1:ppn=1,pmem=5000mb -l walltime=0:02:00:00
                # ./eval_single_pipeline.sh -pl pipeline.pickle -d data.pickle

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
                        -d | --data)   	    shift
                                            DATA_FILE=${1}
                                            ;;
                    * ) echo "illegal ARGUMENT ${1}"
                    esac
                    shift
                done

                echo "Loading singularity"
                module load tools/singularity/3.5

                # work in the nemo shared dir
                cd <NEMO_SHARED_DIR>
                # cd <SINGULARITY_CONTAINER_DIR>              # work somewhere outside home, else singularity will not be able to access paths outside from home

                # Note: make sure that the container is executeable
                singularity exec --bind <WS_DIR>:<WS_DIR_LOCAL> <SINGULARITY_CONTAINER> python -c "import dill; from <__main__> import *; pl=dill.load(open('$PIPELINE_FILE', 'rb')); data=dill.load(open('$DATA_FILE', 'rb')); pl.eval(data)"
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

    # pipelines are unique --> required for 1 to 1 logging
    pl_hashes = [(pl, pl.__hash__(), pl._name) for
                 pl in pdata.get_by_name(pl_container).data]
    print(pl_hashes)

    non_uniques = [plh for plh in set(pl_hashes) if pl_hashes.count(plh) > 1]
    names = [plh[2] for plh in pl_hashes]
    non_unique_n = [n for n in set(names) if names.count(n) > 1]
    assert non_uniques == [], "Pipelines in list have to be unique -> copy "\
        f" and change the names for: {non_uniques}"
    assert non_unique_n == [], "Pipeline names have to be unique -> copy and"\
        f" change names for: {non_unique_n}"

    return pdata


def attach_log_files(pdata):
    """ make sure each pipeline is logging to the log dir in config """

    conf = pdata.get_by_name('nemo_config').data
    pls = pdata.get_by_name('pipelines').data

    nemo_log_files = []
    local_log_files = []
    for pl in pls:
        pl._log_eval = True
        fname = pl._name.replace(' ', '_') + '.log'
        nemo_fpath = conf['nemo_log_dir'].joinpath(fname)
        pl._logger = PlainLogger(nemo_fpath)
        nemo_log_files.append(nemo_fpath)
        local_log_files.append(conf['local_log_dir'].joinpath(fname))

    pdata.get_by_name('pipelines').meta['nemo_logs'] = nemo_log_files
    pdata.get_by_name('pipelines').meta['local_logs'] = local_log_files

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

    assert ssh_client is not None, "Could not establish ssh connection to "\
        "nemo - is vpn running?"
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

    # make sure local temp exists
    conf['local_pickle_tmp'].mkdir(exist_ok=True, parents=True)

    # Use the hash for identification
    pipeline_files = []
    data_files = []
    for pl in upls:
        fname = f'pipeline_{pl.__hash__()}.pickle'
        dill.dump(pl, open(conf['local_pickle_tmp'].joinpath(fname), 'wb'))
        pipeline_files.append(fname)
    for ud in udata:
        fname = f'data_{ud.__hash__()}.pickle'
        dill.dump(ud, open(conf['local_pickle_tmp'].joinpath(fname), 'wb'))
        data_files.append(fname)

    return {'pipeline_files': pipeline_files, 'data_files': data_files}


def transfer_pickles_local_to_nemo(ssh_client, conf, file_dict):
    NEMO_EVAL_LOGGER.info("Transfering pickle files")
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
    open(script_local_tmp_fl, 'w').write(
        inspect.cleandoc(
            get_eval_sh()
            .replace('<SINGULARITY_CONTAINER>',
                     str(conf['singularity_container_dir'].joinpath(
                         conf['singularity_container_name'])
                         ))
            .replace('<NEMO_SHARED_DIR>',
                     str(conf['nemo_shared_dir']))
            .replace('<SINGULARITY_CONTAINER_DIR>',
                     str(conf['singularity_container_dir']))
            .replace('<__main__>',
                     str(conf['__main__'].stem))
            .replace('<WS_DIR>', str(conf['data_root_nemo']))
            .replace('<WS_DIR_LOCAL>', str(conf['data_root_local']))
        )
    )

    ftp_client.put(str(script_local_tmp_fl),
                   str(script_remote_fl))

    ssh_client.exec_command(f'chmod 755 {script_remote_fl}')

    ftp_client.close()


def transfer_pipeline_dependencies(pdata):

    NEMO_EVAL_LOGGER.info("Transfering necessary modules")

    conf = pdata.get_by_name('nemo_config').data
    needed_modules = [conf['__main__']] + conf['script_paths_to_copy']

    # use rsync as we want to glob for *.py
    cmd = (f'rsync -Pavz --exclude=".*" --include="*/" --include="*.py" '
           f'--exclude="*" -e "ssh -i {conf["ssh_key_file"]}" <src_dir>'
           f' fr_md1104@login1.nemo.uni-freiburg.de:{conf["nemo_shared_dir"]}'
           )

    for pth in needed_modules:
        os.system(cmd.replace('<src_dir>', str(pth)))


def send_data_to_nemo(pdata):

    conf = pdata.get_by_name('nemo_config').data
    ssh_client = pdata.get_by_name('ssh_client').data

    # create the folders
    prepare_for_transfer(conf, ssh_client=ssh_client)

    # pack at local folder and return a map
    file_dict = pack_pipelines_and_data_to_pickle(pdata)

    # transfer pickles
    transfer_pickles_local_to_nemo(ssh_client, conf, file_dict)

    # also transfer any module dependencies
    transfer_pipeline_dependencies(pdata)

    return pdata


def initialize_ssh_connection_to_nemo(pdata, trg_container='ssh_client'):
    conf = pdata.get_by_name('nemo_config').data
    trg = pdata.get_by_name(trg_container, create_if_missing=True)
    ssh_client = start_connection_to_nemo(conf)

    # now make sure that nemo and local log dirs exists
    pls_meta = pdata.get_by_name('pipelines').meta
    nemo_log_dir = str(pls_meta['nemo_logs'][0].parent)
    _, _, _ = ssh_client.exec_command(f'mkdir -p {nemo_log_dir}')
    pls_meta['local_logs'][0].parent.mkdir(exist_ok=True, parents=True)

    trg.data = ssh_client

    return pdata


def send_eval_jobs_to_nemo(pdata, jobs_container='jobs'):
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

    jobs = []

    for pl_f, data_f in zip(pls_files, data_files):
        job_id = start_job(ssh_client, conf, pl_f, data_f)
        jobs.append(job_id)

    NEMO_EVAL_LOGGER.info(f"Started {len(jobs)} with ids: {jobs}")
    trg = pdata.get_by_name(jobs_container, create_if_missing=True)
    trg.data = jobs

    return pdata


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
    script_flags = f'-pl {pl_file} -d {data_file}'

    q2use = '' if conf['queue_to_use'] == '' else '-q ' + conf['queue_to_use']

    stdin, stdout, stderr = ssh_client.exec_command(
        'msub '  # '-q express'
        f'{q2use} '
        f'-o {log_file} -e {log_file} '
        f'-v SCRIPT_FLAGS="{script_flags}" '
        f'-l nodes=1:ppn=1,pmem={conf["ram_to_use"]},'
        f'walltime={conf["time_to_allocate"]} {eval_script_path}'
    )

    id_str = stdout.read().decode('ascii').replace('\n', '')

    # TODO: Check stderr

    if id_str == '':
        job_id = -1                     # something odd happened
    else:
        job_id = int(id_str)

    return job_id


def mirror_log_files(pdata):
    """ Use tail -f to pipe from remote to local log files """

    conf = pdata.get_by_name('nemo_config').data
    pls_meta = pdata.get_by_name('pipelines').meta
    cmd_template = (f"ssh -i {conf['ssh_key_file']} {conf['nemo_user']}"
                    f"@{conf['nemo_host']} tail -f <remote_log_file>")

    ssh_client = pdata.get_by_name('ssh_client').data
    procs = []

    for nemo_log_file, local_log_file in zip(
            pls_meta['nemo_logs'], pls_meta['local_logs']):

        # touch files to make sure they are there if no print has happened yet
        # e.g because job is still in queue
        ssh_client.exec_command(f"touch {nemo_log_file}")
        local_log_file.parent.mkdir(parents=True, exist_ok=True)
        local_log_file.touch()

        # Start mirroring processes
        procs.append(
            Popen(cmd_template
                  .replace('<remote_log_file>', str(nemo_log_file))
                  .split(" "),          # popen requires a list of args
                  stdout=open(local_log_file, 'w')
                  )
        )

    trg = pdata.get_by_name('tail_procs', create_if_missing=True)
    trg.data = procs

    return pdata


def stop_mirror_log_files(pdata):
    for p in pdata.get_by_name('tail_procs').data:
        p.terminate()

    return pdata


def start_monitoring(pdata):
    """ Check the log files for the jobs by continously monitoring them
    over ssh via tail -f
    """
    pls_c = pdata.get_by_name('pipelines')
    logfiles = pls_c.meta['local_logs']
    job_ids = pdata.get_by_name('jobs').data
    names = [f.stem + '_' + str(id) for f, id in zip(logfiles, job_ids)]
    monitor = PipelineMonitor(logfiles=logfiles, names=names)

    # This will run until all pipelines are finished (last step in logs)
    # or manual interupt is sent
    monitor.show()

    return pdata


def clean_tmp_files(pdata):
    conf = pdata.get_by_name('nemo_config').data
    ssh_client = pdata.get_by_name('ssh_client').data

    # local cleaning
    local_tmp = conf['local_pickle_tmp']

    if local_tmp.joinpath('nemo_eval').exists():
        for f in local_tmp.joinpath('nemo_eval').glob('*'):
            f.unlink()

    local_tmp.joinpath('nemo_eval.log').unlink(missing_ok=True)

    for f in (list(local_tmp.rglob('data*.pickle'))
              + list(local_tmp.rglob('pipeline*.pickle'))):

        f.unlink()

    # remote cleaning
    stdin, stdout, stderr = ssh_client.exec_command(
        f'rm {conf["nemo_log_dir"]}/*.log'
    )

    return pdata


def test_print(pdata):
    print(pdata.get_containers())
    return pdata


def test_sleep(pdata):
    sleep(30)
    return pdata

# TODO: Unfortunately, even dill is not capeable of getting the functions
# with import right ---> here we fail at sleep
# (or also with basic `import time` at time.sleep) as this is unknown in
# the singularity container loading the pipeline via dill.load()
#
# Think about a proper way of sourcing....
# Find a way of sourcing the file where the pls are defined --> e.g. copy
# then source everything via: from main import *


# ==============================================================================
# Pipeline
# ==============================================================================
eval_pl = xPipeline('eval_on_nemo_pl', log_eval=False)
eval_pl.add_step(('Check the input', validate_data, {}))
eval_pl.add_step(('Attach logs', attach_log_files, {}))
eval_pl.add_step(('Initialize ssh connection',
                  initialize_ssh_connection_to_nemo, {}))
# Not we clean up to remove residue from the temp dir which might be there
# from a previous run. -> don't clean after, in case there might be something
# which needs to be looked up
eval_pl.add_step(('Clean tmp files', clean_tmp_files, {}))
eval_pl.add_step(('Send data to nemo', send_data_to_nemo, {}))
eval_pl.add_step(('Eval jobs on nemo', send_eval_jobs_to_nemo, {}))
eval_pl.add_step(('Mirror log files', mirror_log_files, {}))
eval_pl.add_step(('Start monitoring', start_monitoring, {}))
eval_pl.add_step(('Stop mirroring', stop_mirror_log_files, {}))

stop_pl = xPipeline('cleanup_pl', log_eval=False)
stop_pl.add_step(('Stop mirroring', stop_mirror_log_files, {}))
stop_pl.add_step(('Clean tmp files', clean_tmp_files, {}))

if __name__ == '__main__':

    testpl = xPipeline('testpipeline',)
    testpl.add_step(('testprint', test_print, {}))
    testpl.add_step(('testsleep', test_sleep, {}))
    testpl.add_step(('testprint 2', test_print, {}))

    testpl2 = xPipeline('testpipeline2',)
    testpl2.add_step(('testprint', test_print, {}))
    testpl2.add_step(('testsleep', test_sleep, {}))
    testpl2.add_step(('testprint 2', test_print, {}))

    testpl3 = xPipeline('testpipeline3',)
    testpl3.add_step(('testprint', test_print, {}))
    testpl3.add_step(('testsleep', test_sleep, {}))
    testpl3.add_step(('testprint 2', test_print, {}))

    testdt = xPData([1, 2, 3], name='testdata')
    testdt2 = xPData([1, 2, 3, 4], name='testdata')

    pdata = xPData(
        [
            xPData(get_default_config(), name='nemo_config'),
            # NOTE: Pipelines have to be unique (at least different names)
            # as this is required for monitoring the remote logging state
            xPData([testpl,
                    testpl2,
                    # testpl3
                    ],
                   name='pipelines'),
            xPData([testdt,
                    testdt2,
                    # testdt
                    ],
                   name='data'),
        ],
        name='eval_on_nemo_data'
    )

    eval_pl.eval(pdata)
    # stop_pl.eval(pdata)           # if eval_pl fails to finish
