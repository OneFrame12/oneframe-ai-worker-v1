"""
OneFrame V57 — Config V17
"""
from dataclasses import dataclass, field
from typing import List, Optional
import json
import os


@dataclass
class YOLOConfig:
    model_path: str = "oneframe_v3_best.pt"
    target_classes: Optional[List[int]] = None
    confidence: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    half: bool = True
    max_det: int = 30
    min_ball_size_px: int = 2
    max_ball_size_px: int = 80
    frame_skip: int = 2
    use_sahi: bool = False
    sahi_slice_size: int = 640
    sahi_overlap: float = 0.2
    sahi_min_movement_px: int = 0
    lens_distortion_correction: bool = False
    detect_classes: list = field(default_factory=lambda: ["ball", "person"])


@dataclass
class RFDETRConfig:
    model_path: str = ""
    conf_threshold: float = 0.25
    device: str = "cuda"
    sports_ball_class_id: int = 37
    ball_class_name: str = "ball"
    min_ball_size_px: float = 4.0
    max_ball_size_px: float = 90.0
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 2.2
    max_jump_px: float = 280.0
    kalman_max_distance_px: float = 320.0
    motion_state_max_age_sec: float = 2.0
    max_jump_px_per_sec: float = 900.0
    min_jump_gate_px: float = 120.0
    yolo_agreement_center_px: float = 25.0
    yolo_agreement_iou: float = 0.25
    stale_high_conf_threshold: float = 0.70
    yolo_fallback_enabled: bool = False


@dataclass
class PlayerDetectionConfig:
    enabled: bool = True
    confidence: float = 0.5
    detection_interval: int = 15
    min_player_size_px: float = 20.0
    max_player_size_px: float = 400.0
    min_players_for_clustering: int = 4
    jersey_region_ratio: float = 0.4
    high_pressure_threshold: int = 3


@dataclass
class TrainingFrameExportConfig:
    enabled: bool = True
    confidence_threshold: float = 0.30
    max_frames_per_match: int = 50
    export_missed_detections: bool = True
    max_missed_exports_per_match: int = 100
    min_gap_sec: float = 2.0
    jpeg_quality: int = 92


@dataclass
class KalmanConfig:
    process_noise: float = 150.0
    measurement_noise: float = 20.0
    max_prediction_frames: int = 120
    confirm_after_frames: int = 2


@dataclass
class TrackingConfig:
    base_max_pixel_jump: float = 250.0
    adaptive_jump_factor: float = 2.0
    absolute_max_pixel_jump: float = 400.0
    distance_weight: float = 0.6
    confidence_weight: float = 0.4
    gk_kick_zone_bottom_pct: float = 0.85
    gk_kick_min_speed: float = 250.0
    filter_gk_kicks: bool = True
    process_every_n_frames: int = 1
    target_fps: Optional[float] = None
    max_frames_lost: int = 8


@dataclass
class AudioConfig:
    sta_window_sec: float = 0.5
    lta_window_sec: float = 10.0
    trigger_ratio: float = 3.0
    min_peak_separation_sec: float = 2.0
    normalize: bool = True


@dataclass
class ClipConfig:
    pre_event_sec: float = 14.0
    long_range_pre_event_sec: float = 18.0
    post_event_sec: float = 8.0
    min_clip_duration_sec: float = 10.0
    max_clip_duration_sec: float = 35.0
    merge_distance_sec: float = 8.0


@dataclass
class DangerZoneConfig:
    incoming_min_speed: float = 250.0
    min_frames_in_zone: int = 3
    min_event_score: float = 0.50
    require_approaching: bool = True
    long_range_detection_radius: float = 2.5
    long_range_min_speed: float = 500.0
    gk_origin_zone_pct: float = 0.40


@dataclass
class VelocityClassification:
    """Rangos de velocidad en px/s para clasificar tipo de acción."""

    possession_max: float = 150.0
    short_pass_max: float = 400.0
    long_pass_max: float = 650.0
    shot_max: float = 1200.0
    # > shot_max = artifact / false positive


@dataclass
class EventScoringConfig:
    speed_weight: float = 0.25
    audio_weight: float = 0.15
    direction_weight: float = 0.20
    zone_weight: float = 0.05
    trajectory_weight: float = 0.10
    player_zone_weight: float = 0.25
    min_clip_score: float = 0.20
    dead_ball_multiplier: float = 0.15
    restart_multiplier: float = 0.45
    reference_speed: float = 600.0
    audio_search_window_sec: float = 3.0
    trajectory_lookback_frames: int = 15
    max_moments_per_sequence: int = 2
    min_moment_separation_sec: float = 3.0
    sequence_window_sec: float = 12.0
    max_approach_angle_deg: float = 45.0


@dataclass
class CameraCalibrationConfig:
    """
    Coeficientes de calibración para corregir distorsión de gran angular.
    Se obtienen una vez con cv2.calibrateCamera() usando un patrón de ajedrez.
    Si no hay calibración, se usa un set genérico para GoPro Hero 12 Wide 4K.
    """
    fx: float = 1400.0
    fy: float = 1400.0
    cx: float = 1920.0
    cy: float = 1080.0
    k1: float = -0.28
    k2: float = 0.08
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    enabled: bool = False
    expected_width: int = 3840
    expected_height: int = 2160


@dataclass
class VisionConfig:
    detector_mode: str = "yolo_primary"
    rfdetr: RFDETRConfig = field(default_factory=RFDETRConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)
    danger_zone: DangerZoneConfig = field(default_factory=DangerZoneConfig)
    velocity: VelocityClassification = field(default_factory=VelocityClassification)
    scoring: EventScoringConfig = field(default_factory=EventScoringConfig)
    player_detection: PlayerDetectionConfig = field(default_factory=PlayerDetectionConfig)
    training_frames: TrainingFrameExportConfig = field(default_factory=TrainingFrameExportConfig)
    camera: CameraCalibrationConfig = field(default_factory=CameraCalibrationConfig)

    @classmethod
    def from_json(cls, path: str) -> "VisionConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as file_stream:
            data = json.load(file_stream)
        config = cls()
        for section_name, section_data in data.items():
            section = getattr(config, section_name, None)
            if section is not None and not isinstance(section_data, dict):
                setattr(config, section_name, section_data)
                continue
            if section is not None and isinstance(section_data, dict):
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        return config

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)


DEFAULT_CONFIG = VisionConfig()


# Legacy compat
@dataclass
class _GameConfigCompat:
    CAMERA_TYPE: str = "arco_norte"
    STA_WINDOW_SEC: float = DEFAULT_CONFIG.audio.sta_window_sec
    LTA_WINDOW_SEC: float = DEFAULT_CONFIG.audio.lta_window_sec
    STA_LTA_THRESHOLD: float = DEFAULT_CONFIG.audio.trigger_ratio
    MIN_CLIP_INTERVAL_SEC: float = DEFAULT_CONFIG.clips.merge_distance_sec
    CLIP_PRE_GOL_SEC: float = DEFAULT_CONFIG.clips.pre_event_sec
    CLIP_POST_GOL_SEC: float = DEFAULT_CONFIG.clips.post_event_sec
    CLIP_PRE_PELIGRO_SEC: float = DEFAULT_CONFIG.clips.pre_event_sec
    CLIP_POST_PELIGRO_SEC: float = DEFAULT_CONFIG.clips.post_event_sec


@dataclass
class _StorageConfigCompat:
    OUTPUT_CLIPS_DIR: str = "/tmp/clips"


@dataclass
class _ProcessingConfigCompat:
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MATCH_ID: str = "unknown"
    VIDEO_FILES: List[str] = field(default_factory=list)


vision_cfg = DEFAULT_CONFIG
game_cfg = _GameConfigCompat()
storage_cfg = _StorageConfigCompat()
processing_cfg = _ProcessingConfigCompat()
calibracion_data = None


def validate_config():
    return None


def get_clip_timing(event_type):
    if str(event_type).lower() == "goal":
        return game_cfg.CLIP_PRE_GOL_SEC, game_cfg.CLIP_POST_GOL_SEC
    return game_cfg.CLIP_PRE_PELIGRO_SEC, game_cfg.CLIP_POST_PELIGRO_SEC
