# OreoML [![Join the chat at https://gitter.im/Oreoweb/community](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/Oreoweb/community) ![test](https://forthebadge.com/images/badges/made-with-python.svg)
<p align="center">
    <img src="https://github.com/harishsg99/OreoML/blob/master/terminal.gif?raw=true">
</p>

## How to Use

OreoML can be installed [via pip](https://pypi.org/project/OreoML/):

Note : Rename the your target column as target in your dataset to predict or classify using OreoML tool
```shell
pip3 install OreoML
```
For Training Model
```shell
oreoml_train --train-csv titanic.csv --model-dir /tmp/model.pkl --mode regression
```

For Prediction Model
```shell
oreoml_predict --train-csv train_titanic.csv --test-csv test_titanic.csv --model-dir /tmp/model.pkl
```


You may also invoke OreoML directly from Python. (e.g. via a Jupyter Notebook)

```python
from OreoML import train

automl_train('titanic.csv', '/tmp/model.pkl','regression')
```
```python
from OreoML import predict

automl_predict('train_titanic.csv', 'test_titanic.csv','/tmp/model.pkl')
```
