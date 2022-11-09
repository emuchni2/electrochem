from setuptools import setup

setup(
    name='electrochem',
    version='0.1.0',    
    description='python code for electrochem',
    url='https://github.com/emuchni2/electrochem',
    author='Ethan Muchnik',
    author_email='ethan.k.muchnik@gmail.com',
    license='NA',
    packages=['electrochem'],
    install_requires=['bokeh',
                      'eclabfiles',                     
                      'scipy',
                      'numpy',
                      'pandas'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)