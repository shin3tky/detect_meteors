"""Write per-pair evidence alongside the usual candidate files."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from meteor_core.outputs.file_handler import FileOutputConfig, FileOutputHandler
from meteor_core.schema import DetectionResult


@dataclass
class PairReportConfig(FileOutputConfig):
    report_path: str = ""


class PairReportHandler(FileOutputHandler):
    """Record successful pairs through the existing output lifecycle."""

    plugin_name = "pair_report"
    name = "Experimental pair report"
    version = "0.1.0"
    ConfigType = PairReportConfig

    def __init__(self, config: PairReportConfig) -> None:
        if not config.report_path:
            raise ValueError("pair_report requires a new report_path")
        self.report_path = Path(config.report_path)
        super().__init__(config)
        self.report_path.touch(exist_ok=False)

    def on_detection_result(
        self, context: dict[str, Any], result: DetectionResult, filepath: str
    ) -> None:
        metadata = context["metadata"]
        row = {
            "current": Path(filepath).name,
            "frame_index": metadata["frame_index"],
            "prev_frame_index": metadata["prev_frame_index"],
            **result.to_dict(),
        }
        with self.report_path.open("a", encoding="utf-8") as report:
            report.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
