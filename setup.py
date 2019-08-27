from setuptools import setup, find_packages

setup(
    name='MyPackageName2',
    version='1.0.0',
    url='https://github.com/JPatrick9793/Pairs_Trading.git',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
