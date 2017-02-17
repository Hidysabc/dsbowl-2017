
from distutils.core import setup

setup(
    name='hwimg',
    version='0.01',
    author=['Hidy Chiu', 'Wei-Yi Cheng'],
    author_email=['hidy0503@gmail.com','ninpy.weiyi@gmail.com'],
    packages=['hwimg'],
    include_package_data = True,
    entry_points={
        'console_scripts': [
            'preprocess=hwimg.preprocessing:main'
        ]
    },
    url='http://hidysabc.com',
    license='LICENSE.txt',
    description='Image processing libraries for Data Science Bowl 2017 challenge',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.11.1",
        "pandas >= 0.18.1",
        "pydicom >= 0.9.9",
        "scipy >= 0.18.1"
    ],
)

