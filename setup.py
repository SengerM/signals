import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name = "signals",
	version = "0",
	author = "Matias Senger",
	author_email = "m.senger@hotmail.com",
	description = "Utilities for signal analysis",
	long_description = long_description,
	long_description_content_type = "text/markdown",
	url = "https://github.com/SengerM/signals",
	packages = setuptools.find_packages(),
	classifiers = [
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
