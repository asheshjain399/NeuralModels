from setuptools import setup, find_packages    

setup(
	name='neuralmodels', 
	version='0.0.1',
	packages=find_packages(),
	license="MIT License (See LICENSE)",
	author="Ashesh Jain",
	author_email="ashesh@cs.cornell.edu",
	install_requires=[
	"numpy >= 1.8.1",
	"Theano >= 0.6.0",
	],
)
