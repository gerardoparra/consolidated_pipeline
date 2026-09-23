"""Process-local compatibility fixes for the legacy Anipose CLI.

Python imports a module named ``sitecustomize`` during interpreter startup.
The pipeline adds this directory to PYTHONPATH only for ``anipose filter``.
"""

from __future__ import annotations

import inspect

import pandas as pd


_to_hdf = pd.DataFrame.to_hdf
_key = inspect.signature(_to_hdf).parameters.get("key")

if _key is not None and _key.kind is inspect.Parameter.KEYWORD_ONLY:

    def _to_hdf_with_positional_key(self, path_or_buf, *args, **kwargs):
        if args:
            if "key" in kwargs:
                raise TypeError("to_hdf() received the HDF key twice")
            kwargs["key"] = args[0]
            args = args[1:]
        if args:
            raise TypeError(
                "Only the legacy positional HDF key is supported; "
                "pass remaining to_hdf arguments by keyword"
            )
        return _to_hdf(self, path_or_buf, **kwargs)

    pd.DataFrame.to_hdf = _to_hdf_with_positional_key
