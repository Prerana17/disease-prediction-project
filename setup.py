from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README.md for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='disease-analysis-cli',
    version='0.1.0',
    packages=find_packages(where="disease-analysis-cli/src"),
    package_dir={'': 'disease-analysis-cli/src'},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'streamlit',
        'click',
        'pytest',
        'joblib',
    ],
    author='Prerana Chandratre',
    author_email='cprerana17@gmail.com',
    description='A CLI and Streamlit-based tool for EDA and disease prediction using patient profile data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    entry_points={
        'console_scripts': [
            'disease-cli = cli:cli'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    project_urls={
        'Repository': 'https://gitlab.hzdr.de/your-username/group-3-disease-analysis-cli',
        'Documentation': 'https://gitlab.hzdr.de/your-username/group-3-disease-analysis-cli/-/wikis',
    },
    python_requires='>=3.8',
)
