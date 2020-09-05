# OreoML [![Join the chat at https://gitter.im/Oreoweb/community](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/Oreoweb/community) ![test](https://forthebadge.com/images/badges/made-with-python.svg)
<p align="center">
    <img src="https://github.com/harishsg99/OreoML/blob/master/OreoML.png?raw=true">
</p>

## How to Use

OreoML can be installed [via pip](https://pypi.org/project/OreoML/):

```shell
pip3 install OreoML
```
For Training Model
```shell
oreoml_train titanic.csv /tmp/model.pkl regression
```

For Prediction Model
```shell
oreoml_train train_titanic.csv test_titanic.csv /tmp/model.pkl
```


You may also invoke OreoML directly from Python. (e.g. via a Jupyter Notebook)

```python
from OreoML import train

automl_train('titanic.csv', '/tmp/model.pkl','regression')
```
```python
from OreoML import predict

automl_train('train_titanic.csv', 'test_titanic.csv','/tmp/model.pkl')
```
