from setuptools import setup, find_packages

setup(
    name="malayalam-bpe-tokenizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.0.0",
        "torch>=1.7.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Malayalam BPE Tokenizer for Hugging Face Transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/malayalam-bpe-tokenizer",
) 