# timestamp-reconstructor

## Description
A python tool for process mining that reconstructs timestamps of missing events in an event log


## Installation
1. You need a working instance of Matlab (https://mathworks.com/products/matlab.html)

2. Create a virtual environment or use an existing one.

3. Install all needed requirements:

`pip install -r requirements.txt`

## Run the tool

Run the tool by using the following command:

`python main.py -m 2 -t 5`

The parameter -m chooses a built-in model.
The parameter -t chooses a built-in trace.

In a future version, it will be possible to dynamically import existing models and traces via files.