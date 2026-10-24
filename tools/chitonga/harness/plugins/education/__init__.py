"""The education plug-in (Grade 1 v0). Data: internal/chitonga_train/assistant_library/plugin_education/ (built by
build_plugin.py); code: hooks.py (the language and marking routes)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "internal", "chitonga_train",
                                     "assistant_library", "plugin_education"))


def load(data_dir=None):
    from core.plugin import Plugin  # noqa: PLC0415
    from .hooks import Hooks  # noqa: PLC0415
    hooks = Hooks()
    return Plugin(data_dir or DATA, hooks=hooks), hooks
