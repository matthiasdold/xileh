#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This module contains a general pipeline data class
# which should be used to generalize the data structure
# that is passed between processing steps and can be used
# for DQ checks

import numpy as np

from logging import Logger
from xileh.utils.logger import xileh_log_this


class xPData(object):

    """
    The generalized pipeline data container
    """

    def __init__(self, data=None, header={},
                 meta={}, name=None, logger=None):
        """ Create the xPData object with:

        Parameters
        ----------

        data : numpy array or array like,
            data which can be expressed as a n-tensor,
            usually of the form (n_variables x n_times x ...)

            Special usage:
            Use a list of xPData object within on xPData object to pass
            multiple data entities through a pipeline

            Note: More involved data types are possible, such as mne.raw
            or epoch object. In this case you either have to overwrite
            all functions to be used in a pipeline by overwriting the
            getter accordingly

        header : dict, optional
            general description of the data, potentially containing
            information to controll the processing flow (flags)
            always requires a "name": <container_name> key value pair
            at least
        meta : dict, optional
            meta data which is linked to the data as a whole or
            individual dimensional subsets. The dimesions have to be
            aligned to the data dimension. E.g. if data is
            (n_variables x n_times) and we would have meta properties
            per time recording accross all varibales, the meta element
            should be key : array.shape(1, n_times)
        name : str
            A container name. This is used as syntactical sugar for the constructor.
            It will always create a field 'name' in header, to keep the logic
            of data, header and meta clean. The 'name' will be kept as an
            attribute only for convenience
        logger : logging.Logger
            a logger for interaction with instances of the class


        """
        self._validate_input(data, header, meta, name)

        # Add name to header -> no overwrite as _validate_input
        # checks for ambiguity
        if name is not None:
            header['name'] = name

        # init the properties
        self._data = None
        self._header = {}
        self._meta = {}
        self.name = ''          # will be updated by the header setter

        # have the setters called on init
        self.data = data
        self.header = header
        self.meta = meta

        self.logger = Logger('default')

    def __repr__(self):
        """ Print more information about the container on repl call """
        return super().__repr__() + f"\nContainer name: {self.header['name']}"

    def _validate_input(self, data, header, meta, name):
        """ Some DQ checks"""

        # At least on name
        assert 'name' in header.keys() or name is not None,\
            "At least a key value pair with key='name' is required in the header"

        # Not in both
        assert not ('name' in header.keys() and name is not None),\
            (f"A header['name'] and a name={name} variable is provided,"
            " please provide only one.")


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        # Use a list checking on names if more containers will be appended
        if isinstance(val, list):
            val = CheckedList(val, self)
        self._data = val

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, hdict):
        if 'name' in hdict.keys():
            self.name = hdict['name']
        self._header.update(hdict)

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, mdict):
        # check if lengths are aligned if data of object with shape is passed
        for k, v in mdict.items():
            if 'shape' in dir(v):
                if not any([v.shape[i] == self._data.shape[i]
                            for i in range(len(v.shape))]):
                    raise ValueError(
                        "Please provide meta data with shape property aligned"
                        " to at least one dimension of the data within this"
                        f" objects data: self.data.shape={self._data.shape}"
                        f" got v.shape={v.shape} for key={k}"
                    )

        self._meta.update(mdict)

    @xileh_log_this()
    def update_meta(self, mdict, logger=Logger('_')):
        # add to meta using the setter with dq check
        # note, the setter will use an update on the dict
        self.meta = mdict

    @xileh_log_this()
    def _check_dict(self, d, key, missing=None, logger=Logger('_'), dname=''):
        """ Check a dictionary and return a default 'missing' value if key
        not in dict.keys()

        Parameters
        ----------
        d : dict
            dict to check
        key : str
            key to look for in
        missing : object, optional
            any object to return in case lookup failed
        logger : logging.Logger, optional
            a logger, usually set by the xileh_log_this wrapper or
            manually provided
        dname : str, optional
            name of the dictionary as a description

        Returns
        -------
            d[key]  if key in d.keys() else missing

        """

        if key in d.keys():
            return d[key]
        else:
            logger.info("No {key} in pdata.{dname}  - continue with {missing}")
            return missing

    @xileh_log_this()
    def check_header(self, key, logger=Logger('_'), missing=''):
        return self._check_dict(self.header, key, logger=self.logger,
                                dname='header', missing=missing)

    @xileh_log_this()
    def check_meta(self, key, logger=Logger('_'), missing=''):
        return self._check_dict(self.meta, key, logger=self.logger,
                                dname='meta', missing=missing)

    def get_by_name(self, name):
        """ Traverse the data container and look for a (sub) container
        with the given name and return it if found

        Parameters
        ----------
        name : str
            name to look up in the data containers header dict

        Returns
        -------
            data : xPData or None
                A data container with the given name or None if no container
                with the given name can be found

        """
        data = None
        if 'name' in self.header.keys() and self.header['name'] == name:
            data = self

        # if we have nested data containers, search recursively and stop at first match
        # Note: it is the coders responsibility to avoid conflicts with potentially
        # multiple containers having the same name in their header property
        elif isinstance(self.data, list):
            for pd in [p for p in self.data if isinstance(p, xPData)]:
                data = pd.get_by_name(name)
                if data is not None:
                    break                                           # early stopping
        else:
            data = None

        return data

    def overwrite(self, new_pdata):
        """ Overwrite data, header and meta of this container by
        referencing to a "new_data" container

        Parameters
        ----------
        new_pdata : xPData
            a new container to overwrite this containers content

        """
        self.data = new_pdata.data
        self.header = new_pdata.header
        self.meta = new_pdata.meta

    def get_containers(self):
        """ Return a dict of data entity names and types """
        name = self.header['name']
        d = {name: type(self.data)}

        # check if we have a list with potential nested xPData structs
        if isinstance(self.data, list) or isinstance(self.data, CheckedList):
            children = []
            for pd in [p for p in self.data if isinstance(p, xPData)]:
                children.append(pd.get_containers())
            d[name] = children

        return d

    def get_container_names(self):
        """ Get all container names """
        names = [self.header['name']]

        # check if we have a list with potential nested xPData structs
        if isinstance(self.data, list) or isinstance(self.data, CheckedList):
            for pd in [p for p in self.data if isinstance(p, xPData)]:
                names += pd.get_container_names()

        return names


class CheckedList(list):

    """ A helper list class for which the append method will be linked
        To an xPData container for checking uniqueness if an xPData
        element is appended
    """

    def __init__(self, vals, xpdata):
        list.__init__(self, vals)
        self.xpdata = xpdata

    def append(self, elm):
        """ Check for name conflict if an xPData elements should be
        appended

        Parameters
        ----------
        elm : object

        """
        if isinstance(elm, xPData):
            assert elm.header['name'] not in self.xpdata.get_container_names(), \
                f"Data container '{self.xpdata}' already containes a container"\
                f" with name '{elm.header['name']}', names need to be unique."

        super(CheckedList, self).append(elm)


if __name__ == "__main__":

    tdata = xPData(
        data=[np.eye(5)],
        header={'name': 'testdata', 'description': 'Some data description'},
        meta={'mean_data[0]': 5}
    )

    tdata.data.append(
        xPData(
            [xPData('a', header={'name': 'deepest'})],
            header={'name': 'nesting'})
    )

    chk_list = CheckedList([], tdata)
    chk_list.append('a')
    assert chk_list == ['a']
    chk_list.append(xPData(None, header={'name': 'deepest'}))
    tdata.data.append(xPData(None, header={'name': 'deepest'}))
