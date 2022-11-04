from setuptools import setup, find_packages


setup(
    name='s-curve-beta',
    version='0.1.0',
    license='Apache 2.0',
    author="Vladimir Grankovsky",
    author_email='vladi@hidoba.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/hidoba/s-curve-beta',
    keywords='Efficient Python implementation of the smoothest S-curve robot motion planner ever',
    install_requires=[
          'numpy',
      ],

)
