"""
AFC Hearing Aid System - Setup
================================
Install:
    pip install .

Run directly:
    python realtime_afc.py
"""
from setuptools import setup, find_packages

setup(
    name='afc_hearing_aid_system',
    version='1.0',
    description='Acoustic Feedback Cancellation + Hearing Aid Amplification',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'pyaudio',
        'pystoi',
    ],
)
