from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='s-curve-beta',
    version='0.1.2',
    license='Apache 2.0',
    author="Vladimir Grankovsky",
    author_email='vladi@hidoba.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/hidoba/s-curve-beta',
    keywords='S-curve, motion planning, motion, robotics, robot, motor',
    install_requires=[
          'numpy',
      ],
    description='Efficient Python implementation of the smoothest S-curve robot motion planner ever',
    long_description=long_description,
    long_description_content_type='text/markdown',    
)
