from skbuild_conan import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "sda_bfc",
    version = "0.1.0",
    description = "SDA-BFC",
    readme = "README.md",
    authors = [
        { "name": "Michael Bilevich", "email": "michaelmoshe@mail.tau.ac.il" },
    ],
    python_requires = ">=3.10",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    conanfile = "./conanfile.txt",
    conan_profile_settings={"compiler.cppstd": "17"},
    install_requires=required,
)
