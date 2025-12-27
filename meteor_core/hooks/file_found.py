"""Hook implementations for file discovery in the detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .base import DataclassHook


@dataclass
class AllowAllFilesFoundConfig:
    """Configuration for AllowAllFilesFoundHook."""


class AllowAllFilesFoundHook(DataclassHook[AllowAllFilesFoundConfig]):
    """Default hook that allows every discovered file."""

    plugin_name = "allow_all_files"
    name = "Allow All Files"
    ConfigType = AllowAllFilesFoundConfig

    def on_file_found(self, filepath: str) -> bool:
        return True
