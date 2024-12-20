from setuptools import setup, find_packages

setup(
    name='classification_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn'
    ],
    description='A library for evaluating classification models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sangram Biplab Manabendra Thakur',
    author_email='sangramaimlds@gmail.com',
    url='https://github.com/sbmthakur/classification_lib',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
