import setuptools

setuptools.setup(
    name="extension_audio_separator",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
    author="rsxdalv",
    description="Audio Separator",
    license="MIT",
    url="https://github.com/rsxdalv/extension_audio_separator",
    project_urls={},
    scripts=[],
    install_requires=[
        "audio-separator",
        "protobuf==4.25.3",  # prevent accidental upgrade
        "onnxruntime-gpu==1.18.1",  # version match for CUDA 11.8
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
