from setuptools import setup

setup(
    name='sofia',
    version='0.1',
    packages=['sofia'],
    url='https://github.com/pysofia/SoFIA',
    license='GPLv3',
    author='Anabel del Val',
    description='Sobol-based sensitivity analysis, Forward and Inverse uncertainty propagation with Application to high temperature gases.',
	install_requires=['matplotlib', 'numpy','scipy','sklearn','seaborn']
)
