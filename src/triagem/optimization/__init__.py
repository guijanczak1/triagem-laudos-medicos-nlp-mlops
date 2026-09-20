"""Latency optimization: ONNX export (T20) and benchmark (T21) for the baseline model.

No eager re-exports here on purpose: ``onnx_export.py`` is run as
``python -m triagem.optimization.onnx_export`` (see ``training/__init__.py``
and ``data/__init__.py`` for the same convention), and importing it from this
``__init__`` as well as executing it as ``__main__`` triggers Python's
"module found in sys.modules before execution of __main__" RuntimeWarning.
Import from ``triagem.optimization.onnx_export`` directly instead.
"""
