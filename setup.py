"""Setup configuration for the Phi-2 fine-tuning package."""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read version from __init__.py
with open('src/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            break

setup(
    name='phi2-finetuning',
    version=version,
    description='Fine-tuning Phi-2 with LoRA for humorous responses',
    author='Laurent-Philippe Albou',
    author_email='lpalbou@gmail.com',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)