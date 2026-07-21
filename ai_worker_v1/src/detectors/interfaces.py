from typing import Any, Callable, List, Optional, Protocol, Tuple


DetectionFactory = Callable[..., Any]
DetectionList = List[Any]
DetectorResult = Tuple[DetectionList, DetectionList]


class DetectorAdapter(Protocol):
    name: str
    frame_exporter: Optional[Any]

    @property
    def is_available(self) -> bool:
        ...

    def detect(
        self,
        frame: Any,
        use_sahi: bool = False,
        match_id: str = "unknown",
        frame_num: Optional[int] = None,
    ) -> DetectorResult:
        ...

    def detect_ball(self, frame: Any) -> DetectionList:
        ...

    def get_stats(self) -> dict:
        ...
