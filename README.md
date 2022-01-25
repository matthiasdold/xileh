# Xileh
Module for optimizing pipeline with genetic algorithms
# General structure
The idea of this is to abstract complex processing pipelines based of the `scipy.pipeline` setup,
but with extended data capacities going beyond a feature matrix `X` and potential labels / regression
targets `y`.
For this end, the abstraction is based of two main components, the `pipeline` and the `pipelinedata`.
The pipeline consists of functions acting upon the pipeline data. Functions are added to the pipeline
during initialization and the pipeline is later evaluated given a `pipelinedata` entity.

# Schematic
![schematic](assets/schematic.png)

## Installation
Install to your pip repo or just link, depeding on your liking.
```
pip install .
# or 
pip install develop
```

## Pipelinedata
The `xileh.core.pipelinedata.xPData` implements the pipelinedata container, which contains of three elements:
1.  A data entity, usually an array or array-like. Possible also a list of other `xPData` entities.
2.  Header information as a dictionary containing information about the whole data set used to describe the data in 1.
3.  Meta information as a dictionary containing meta data about 1., such as configuration info for the processing, aggregated measure, or per record meta data. Note that if a dictionary value has a shape property, it is assumed that it should contain a per record (i.e. 1 meta elements per value) info and is thus checked for its length upon initialization. So to provide an array which does not match the length of 1. you have to pack it into an object without shape property first.

All 1.-3. are potentially extended by processing steps within the pipeline. However, processing should in general not change any entities provided during initialization, but create new copies. There might be reasons however, where a change is required (e.g. memory constraint). Overwriting is thus not restricted.

### Modifying data
#### new
#### Add 
#### Delete 

## Pipelines
### Initialization
### Reuse in other modules
### Modification


## Working with xileh
A general example of how a script could be structured


## Data handler
All kudos for the datahandler goes to @dcwil


# TODOS
[ ] Pandas data frame load cycle -> currently loosing column info!
