#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This module contains a general pipeline data class
# which should be used to generalize the data structure
# that is passed between processing steps and can be used
# for DQ checks

# from methodtools import lru_cache

import warnings
import numpy as np

from xileh.utils.datahandler.saving import save_to_folder

# rename so that is is not accidentally loaded
from xileh.utils.datahandler.loading import load_container as _load_container

from copy import deepcopy


class xPData(object):

    """
    The generalized pipeline data container
    """

    def __init__(self, data=None, header={},
                 meta={}, name=None):
        """ Create the xPData object with:

        Parameters
        ----------

        data : numpy array or array like,
            data which can be expressed as a n-tensor,
            usually of the form (n_variables x n_times x ...)

            Special usage:
            Use a list of xPData object within on xPData object to pass
            multiple data entities through a pipeline

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
            A container name. This is used as syntactical sugar for the
            constructor. It will always create a field 'name' in header, to
            keep the logic of data, header and meta clean. The 'name' will be
            kept as an attribute only for convenience


        """
        self._validate_input(data, header, meta, name)

        # Add name to header -> no overwrite as _validate_input
        # checks for ambiguity
        if name is not None:
            # header['name'] = name << this will change the class signature ...
            nheader = header.copy()
            nheader['name'] = name
        else:
            nheader = header.copy()

        # init the properties, these are the 'protected' attrs, which will
        # be retrieved/set via custom getters and setters on e.g. self.data
        self._data = None
        self._header = {}
        self._meta = {}
        self.name = ''          # will be updated by the header setter

        # have the setters called on init
        self.header = nheader           # have header before data to ensure name is set already         # noqa
        self.data = data
        self.meta = meta

    def __getitem__(self, name):
        return self.get_by_name(name)

    def __setitem__(self, name, value):

        if name in ['header', 'meta']:
            print(f">>> You are trying to set an item with name={name}, "
                  "this will most likely fail in the next step as the "
                  "object cannot distinguish between a container with that "
                  "name or the property. If you mean to set a header of meta,"
                  " try using the attribute notation."
                  )

        assert isinstance(value, xPData), "You can only assign xPData"\
            " containers to a xPData container"

        trg = self.get_by_name(name)

        trg.data = value.data

        trg.header = value.header
        trg.header['name'] = name

        trg.meta = value.meta

    def __repr__(self):
        """ Print more information about the container on repl call """
        # return super().__repr__() + f"\nContainer: {self.header['name']}"
        s = f'xPData object at {hex(id(self))} '\
            f'- with size {self.__sizeof__()}\n'
        s += pretty_print_get_containers(self.get_containers())
        return s

    def __setattr__(self, name, value):
        """
        Two possibilities, set another xPData -> register all subcontainers
        or simply overwrite object
        """
        # Assign a container -> overwrite
        if (isinstance(value, xPData) and name in self.get_container_names()):
            trg_c = self[name]
            trg_c.overwrite(value)

            # make sure the old name is preserved, as a renaming with the
            # assignment itself would be a bit counter intuitive
            trg_c.name = name

        # Always set the attr
        # TODO: consider creating new containers via attr setter, i.e.
        # mycont.newcontname = somevalue --> should map to
        # trg = mycont.get_by_name('newcontname'); trg.data = somevalue

        super().__setattr__(name, value)

    def _validate_input(self, data, header, meta, name):
        """ Some DQ checks"""

        # At least on name
        assert 'name' in header.keys() or name is not None,\
            "At least a key value pair with key='name'"\
            " is required in the header"

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
            # start with empty list and append one by one, so that all are
            # registered as attributes
            aux = CheckedList([], self)
            for e in val:
                aux.append(e)

            self._data = aux
        else:
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

        # TODO: Reconsider this restriction / saveguard... unnecessary?
        for k, v in mdict.items():
            if 'shape' in dir(v) and 'shape' in dir(self._data):
                if not any([v.shape[i] == self._data.shape[i]
                            for i in range(len(v.shape))]):
                    raise ValueError(
                        "Please provide meta data with shape property aligned"
                        " to at least one dimension of the data within this"
                        f" objects data: self.data.shape={self._data.shape}"
                        f" got v.shape={v.shape} for key={k}"
                    )

        self._meta.update(mdict)

    def update_meta(self, mdict):
        # add to meta using the setter with dq check
        # note, the setter will use an update on the dict
        self.meta = mdict

    def _check_dict(self, d, key, missing=None, dname=''):
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
        dname : str, optional
            name of the dictionary as a description

        Returns
        -------
            d[key]  if key in d.keys() else missing

        """

        if key in d.keys():
            return d[key]
        else:
            return missing

    def check_header(self, key, missing=''):
        return self._check_dict(self.header, key, dname='header',
                                missing=missing)

    def check_meta(self, key, missing=''):
        return self._check_dict(self.meta, key,
                                dname='meta', missing=missing)

    # @lru_cache(maxsize=16) --> disable as nemo_eval would not work like this. => problem is transport of data as pickle           # noqa
    # TODO: Fix transport not as pickle but as folder generated via datahandler.saver.save()                                        # noqa
    # Not a critical performance overhead as of yet anyways
    def get_by_name(self, name, create_if_missing=False, find_parent=False,
                    parent=None):
        """ Traverse the data container and look for a (sub) container
        with the given name and return it if found

        Parameters
        ----------
        name : str
            name to look up in the data containers header dict
        create_if_missing : bool
            if no container is found, create a new one with the given name
            and append to the self.data if it is a list, else raise Error
        find_parent : bool
            if true return the parent container instead
        parent : xileh.xPData
            a container to store the parent to be returned if find_parent is
            true. Used for recursive call of the method.

        Returns
        -------
            data : xPData or None
                A data container with the given name or None if no container
                with the given name can be found

        """

        data = None
        if self.header['name'] == name:
            data = self
            if find_parent:
                data = parent

        # if we have nested data containers, search recursively and stop at first match         # noqa
        # Note: it is the coders responsibility to avoid conflicts with potentially             # noqa
        # multiple containers having the same name in their header property
        elif isinstance(self.data, list):
            for pd in [p for p in self.data if isinstance(p, xPData)]:
                # create_if_missing only on outer most container
                data = pd.get_by_name(name, find_parent=find_parent,
                                      parent=self)
                if data is not None:
                    break                                     # early stopping

            # full iteration nothing found and last check --> here we create
            if data is None and create_if_missing:
                data = xPData(None, name=name)
                self.data.append(data)

        else:
            if create_if_missing:
                raise ValueError(
                    "Cannot create for type(self.data) != list container")
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

    def gc(self):
        """ Convenience alias for get_containers """
        return self.get_containers()

    def add(self, value, name='', header={}, meta={}):
        """ Convenience alias creating a new container """
        trg = self.get_by_name(name, create_if_missing=True)
        trg.data = value
        trg.header.update(header)
        trg.meta = meta

    def get_container_names(self):
        """ Get all container names """
        names = [self.header['name']]

        # check if we have a list with potential nested xPData structs
        if isinstance(self.data, list) or isinstance(self.data, CheckedList):
            for pd in [p for p in self.data if isinstance(p, xPData)]:
                names += pd.get_container_names()

        return names

    def delete_by_name(self, name):
        """Look for a subcontainer with the given name and drop it"""

        # Sub containers will always be part of a list or the only .data elemnt
        # In both cases it is ok to return a list (potentially empty)

        # make sure it is there
        trg_c = self.get_by_name(name)
        assert trg_c is not None, f"No container with {name=} to delete."

        parent = self.get_by_name(name, find_parent=True)

        if parent.data == trg_c:
            parent.data = []
        elif isinstance(parent.data, list):
            # bypass the setter by using _data here -> else conflict for resetting                                                                       # noqa
            parent._data = CheckedList(
                [c for c in parent.data
                 if isinstance(c, xPData)         # note only containers can be deleted by this approach, no need to check anything else      # noqa
                 and c != trg_c], parent)
        else:
            raise ValueError(f"Parent of target with {name=} has an unknown"
                             " data structure - should either include just"
                             " the container or a list of containers")

        # clear cache to avoid retrieving old data from cache
        # Removed caching on get_by_name -> see above
        # self.get_by_name.cache_clear()

        # attr cleanup
        delattr(parent, name)

    def _to_dict(self):
        """ Transform the container to a dictionary -> for later use of
        storing / serializing the data

        Returns
        -------
        rdict : dict
            dictionary containing the data and reflecting the hierarchy of
            the container
        """

        ddata = {}
        # potentially child containers
        if isinstance(self.data, list):
            ddata['data'] = [v._to_dict() if isinstance(v, xPData) else v
                             for v in self.data]
        else:
            ddata['data'] = self.data

        ddata['header'] = self.header
        ddata['meta'] = self.meta

        rdict = {self.name: ddata, 'datatype': 'xPData'}
        return rdict

    def save(self, fname):
        """ Store the container to a given folder name

        Parameters
        ----------
        fname : str or pathlib.Path
            folder path and name to store the data to
        """

        save_to_folder(self._to_dict(), fname=fname)

    def rename(self, from_name, to_name):
        """ Changes a containers name

        Parameters
        ----------
        from_name : str
            name of the container to rename
        to_name : str
            new name of the container
        """

        src = self.get_by_name(from_name)
        src.header['name'] = to_name

        # also clean the attributes if renaming was not on the outer most container         # noqa
        parent = self.get_by_name(to_name, find_parent=True)
        if parent is not None:
            setattr(parent, to_name, src)
            delattr(parent, from_name)

    def copy(self):
        """
        Create a copy of the container and return it. This method is
        convenience and helps to avoid issues which deepcopy encounters for
        copying nested structures
        """
        ndata = xPData(data=None, name=self.name)
        # potentially child containers

        if isinstance(self.data, list):
            ndata.data = [v.copy() if isinstance(v, xPData) else deepcopy(v)
                          for v in self.data]
        else:
            ndata.data = self.data

        # copy the dictionaries
        ndata.header = self.header.copy()
        ndata.meta = self.meta.copy()

        return ndata


class ContainerNameNotUniqueError(KeyError):
    pass


class CheckedList(list):

    """ A helper list class for which the append method will be linked
        To an xPData container for checking uniqueness if an xPData
        element is appended

        Also registers the name in the parents keys for ['<name>'] access
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
            the object to append

        """

        if isinstance(elm, xPData) and elm.__dict__ != {}:

            if elm.header['name'] in self.xpdata.get_container_names():
                raise ContainerNameNotUniqueError(
                    f"Data container '{self.xpdata.name}' already containes "
                    f"a container with name '{elm.name}', "
                    "names need to be unique.")

            # also register an attribute
            setattr(self.xpdata, elm.name, elm)

        super(CheckedList, self).append(elm)


def from_dict(d):
    """ Given a dictionary representation of an xPData object
    reconstruct --> inverser of xPData._to_dict

    NOTE: The dictionary and its values will be linked to the xPData container
    by reference. Create a deep copy if necessary.

    Parameters
    ----------
    d : dict
        dictionary representation of a xPData object

    Returns
    -------
    xpd : xPData
        the pipeline data instance according to the data in d
    """

    if 'type' in d.keys() and d['type'] == 'xPData':
        # legacy loading
        type_key = 'type'
    else:
        type_key = 'datatype'

    assert d[type_key] == 'xPData', "Dictionary needs to represent a xPData"\
        " object which needs a datatype=='xPData' key:value pair - received"\
        f" {d}"
    names = [k for k in d.keys() if k != type_key]

    assert len(names) == 1, "Unknown structure for casting dict to xPData"\
        " - expected one key for the container name + one 'datatype' "\
        "nothing more"

    elms = d[names[0]]

    # check consistency with header
    if 'name' in elms['header'].keys():
        if elms['header']['name'] != names[0]:
            raise KeyError("Headers name argument not consistent with name"
                           f"provided as a key for the full dict={elms}, "
                           f" name={names[0]}")

    else:
        warnings.warn("Dictionary without 'name' under the 'header' values"
                      " - creating on the fly")
        elms['header']['name'] = names[0]

    # check if we have nested container structure - elements of a list might
    # be containers again. In theory, we could also have containers in dicts
    # which again are in a list or a dict and so forth, but this would also
    # not work for the recursive check of the get_by_name(), -> do not consider
    if isinstance(elms['data'], list):
        for i, elm in enumerate(elms['data']):
            if (isinstance(elm, dict) and type_key in elm.keys()
                    and elm[type_key] == 'xPData'):
                elms['data'][i] = from_dict(elm)

    # add an empty meta if missing --> load from toml instead yaml would cause
    # this
    if 'meta' not in elms.keys():
        elms['meta'] = {}

    return xPData(elms['data'],
                  header=elms['header'],
                  meta=elms['meta'])


def from_container(cpath):
    """ Load a container located at cpath

    Parameters
    ----------
    cpath : str or pathlib.Path
        path to the container

    Returns
    -------
    pdata : xPData
        a pipelinedata container loaded from cpath
    """
    pdata = from_dict(_load_container(cpath))

    return pdata


def pretty_print_get_containers(d, depth=0):
    """ Nicer string representation for return for .get_containers"""
    if isinstance(d, dict):
        s = ''.join([('|   ' * depth + k + ':\t'
                      + pretty_print_get_containers(v, depth=depth + 1))
                     for k, v in d.items()])
    elif isinstance(d, list) and d != []:
        s = '\n' + ''.join([pretty_print_get_containers(e, depth=depth)
                            if isinstance(e, dict) else e
                            for e in d])
        s += '|   ' * (depth - 1) + "'" + '-' * 20 + '\n'
    elif d == []:
        s = 'list' + '\n'
    else:
        s = str(d) + '\n'

    return s


if __name__ == "__main__":

    tdata = xPData(
        data=[np.eye(5)],
        header={'name': 'testdata', 'description': 'Some data description'},
        meta={'mean_data[0]': 5}
    )

    tdata.data.append(
        xPData(
            [xPData('a', header={'name': 'deepest'}),
             xPData(1241, header={'name': 'deepest2'}),
             xPData({'a': 'b'}, header={'name': 'deepest3'}),
             ],
            header={'name': 'nesting'})
    )

    for i in range(5):
        n = tdata.get_by_name(f'test_{i}', create_if_missing=True)
        n.data = '12314'

    tdata.delete_by_name('nesting')

    for i in range(5, 10):
        n = tdata.get_by_name(f'test_{i}', create_if_missing=True)
        n.data = '12314'

    tdata.__dict__

    tdata.data.append(
        xPData('124', name='notsodeep'),
    )

    tdata.data.append(
        xPData([1, 2], name='notsodeep2'),
    )

    tdata_long_list = xPData(
        data=[xPData(i, name=f"some_name_{i}") for i in range(10)],
        header={'name': 'testdata', 'description': 'Some data description'},
        meta={'mean_data[0]': 5}

    )

    chk_list = CheckedList([], tdata)
    chk_list.append('a')
    assert chk_list == ['a']
    chk_list.append(xPData(None, header={'name': 'deepest'}))
    tdata.data.append(xPData(None, header={'name': 'deepest'}))

    # tested with methodtools.lru_cache
    # In [31]: %timeit tdata.get_by_name('testdata')
    # 1.49 µs ± 104 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)        # noqa
    # In [35]: %timeit tdata.get_by_name('deepest')
    # 1.46 µs ± 21.1 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)           # noqa
    # In [47]: %timeit tdata_long_list.get_by_name('some_name_9')
    # 1.4 µs ± 17.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)            # noqa
    # In [51]: %timeit tdata_long_list.get_by_name('some_name_0')
    # 1.4 µs ± 13.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)            # noqa

    # without methodtools.lru_cache
    # In [33]: %timeit tdata.get_by_name('testdata')
    # 360 ns ± 4.96 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)            # noqa
    # In [40]: %timeit tdata.get_by_name('deepest')
    # 2.22 µs ± 39.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)            # noqa
    # ... Most realistic worst case
    # In [44]: %timeit tdata_long_list.get_by_name('some_name_9')
    # 6.21 µs ± 162 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    # ... Most realistic best case
    # In [49]: %timeit tdata_long_list.get_by_name('some_name_0')
    # 1.73 µs ± 14.3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)           # noqa
