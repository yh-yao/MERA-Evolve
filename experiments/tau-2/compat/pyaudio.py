"""Import-only PyAudio shim for tau2 text-mode training.

The tau2 package imports its optional voice modules eagerly. Text experiments
never open audio devices, but still need these names to exist at import time.
"""
paInt8 = 16
paInt16 = 8
paInt32 = 2


class PyAudio:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("PyAudio is unavailable in the verl text-training environment")
