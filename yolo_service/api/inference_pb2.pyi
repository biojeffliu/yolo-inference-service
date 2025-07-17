from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InferVideoRequest(_message.Message):
    __slots__ = ("video_data", "filename")
    VIDEO_DATA_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    video_data: bytes
    filename: str
    def __init__(self, video_data: _Optional[bytes] = ..., filename: _Optional[str] = ...) -> None: ...

class JobResponse(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class GetResultRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class Detection(_message.Message):
    __slots__ = ("class_id", "confidence", "bbox")
    CLASS_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    BBOX_FIELD_NUMBER: _ClassVar[int]
    class_id: int
    confidence: float
    bbox: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, class_id: _Optional[int] = ..., confidence: _Optional[float] = ..., bbox: _Optional[_Iterable[float]] = ...) -> None: ...

class FrameDetections(_message.Message):
    __slots__ = ("detections",)
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    detections: _containers.RepeatedCompositeFieldContainer[Detection]
    def __init__(self, detections: _Optional[_Iterable[_Union[Detection, _Mapping]]] = ...) -> None: ...

class InferenceResult(_message.Message):
    __slots__ = ("frames",)
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    frames: _containers.RepeatedCompositeFieldContainer[FrameDetections]
    def __init__(self, frames: _Optional[_Iterable[_Union[FrameDetections, _Mapping]]] = ...) -> None: ...
