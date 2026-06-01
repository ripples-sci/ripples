
# Copyright (c) Álvaro Cátedra Sánchez <alvaro.catedra.sanchez@gmail.com>,
# unique author and maintainer.

# This module is licensed under the Apache License 2.0. You may use, modify,
# and distribute this software under the terms of the license, provided that
# proper attribution is given to the original author. See LICENSE.txt for
# full details.

from setuptools import setup


# Every value that matters is in `pyproject.toml`, which is the
# single source of truth. This file MUST stay empty of metadata so
# that the two sources can never disagree.

# All metadata is read from pyproject.toml (PEP 621). The call below is
# what setuptools requires to recognise this as a package; it is not
# the place to add metadata.
setup()
