from setuptools import setup

setup(name='gym_graph',
      version='0.0.1',
      packages=['gym_graph', 'gym_graph.envs'],
      install_requires=['gymnasium']  # And any other dependencies the package needs
)
