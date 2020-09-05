from setuptools import setup, find_packages

long_description = '''
Give an input CSV file and a target field you want to predict to OreoML, and get a trained high-performing machine learning.OreoML is an AutoML tool which, offers a *zero code/model definition interface* to getting an high performance machine learning model
'''


setup(
    name='OreoML',
    packages=['OreoML'],  # this must be the same as the name above
    version='1.0.0',
    description='Provide an input CSV and a target field to predict, ' \
    'generate a model.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Harish S.G',
    author_email='harishsg99@gmail.com',
    url='https://github.com/harishsg99/OreoML',
    keywords=['deep learning', 'tensorflow', 'keras', 'automl', 'xgboost'],
    classifiers=[],
    license='MIT',
    entry_points='''
    [console_scripts]
    oreoml_train=OreoML.train:cmd
    oreoml_predict=OreoML.predict:cmd
    ''',
    python_requires='>=3.5',
    include_package_data=True,
    install_requires=['pandas', 'scikit-learn', 'lightgbm', 'tqdm', 'pickle', 'numpy']
)
