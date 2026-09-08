"""Unified result schema and bee individual/group behavior quantification."""

import math
from collections import Counter, defaultdict

import torch

from .pose_motion_tracker import correct_head_tail_sequence


REQUIRED_RECORD_FIELDS = {
    'schema_version', 'frame_index', 'track_id', 'query_index', 'box',
    'keypoints', 'score', 'quality', 'scene_region',
}


def validate_result_schema(records, expected_version=1):
    for index, record in enumerate(records):
        missing = REQUIRED_RECORD_FIELDS - set(record)
        if missing:
            raise ValueError(f'record {index} missing fields: {sorted(missing)}')
        if int(record['schema_version']) != expected_version:
            raise ValueError(f'record {index} has unsupported schema version')
        if record['keypoints'] is not None and tuple(record['keypoints'].shape) != (2, 2):
            raise ValueError(f'record {index} keypoints must have shape [2, 2]')
    return True


def _angle_delta(current, previous):
    return (current - previous + math.pi) % (2 * math.pi) - math.pi


class BeeBehaviorAnalyzer:
    def __init__(self, fps, pixels_per_mm, image_size,
                 heatmap_size=(32, 32), interaction_distance_mm=20.0,
                 entrance_region=None, min_track_frames=3):
        if fps <= 0 or pixels_per_mm <= 0:
            raise ValueError('fps and pixels_per_mm must be positive')
        self.fps = float(fps)
        self.pixels_per_mm = float(pixels_per_mm)
        self.image_size = tuple(image_size)
        self.heatmap_size = tuple(heatmap_size)
        self.interaction_distance_mm = float(interaction_distance_mm)
        self.entrance_region = entrance_region
        self.min_track_frames = int(min_track_frames)
        self.records = []

    def add(self, records):
        validate_result_schema(records)
        self.records.extend(records)

    @staticmethod
    def _center(record):
        box = record['box'].float()
        return (box[:2] + box[2:]) / 2

    @staticmethod
    def _axis_angle(record):
        keypoints = record.get('keypoints')
        if keypoints is None:
            return None
        axis = keypoints[1].float() - keypoints[0].float()
        if float(axis.norm()) < 1e-6:
            return None
        return math.atan2(float(axis[1]), float(axis[0]))

    def _individual(self, track_records):
        track_records = sorted(track_records, key=lambda record: record['frame_index'])
        if len(track_records) < self.min_track_frames:
            return None
        centers = [self._center(record) for record in track_records]
        speeds = []
        angular_velocities = []
        path_length_mm = 0.0
        angles = [self._axis_angle(record) for record in track_records]
        for index in range(1, len(track_records)):
            frame_gap = max(track_records[index]['frame_index'] - track_records[index - 1]['frame_index'], 1)
            distance_mm = float((centers[index] - centers[index - 1]).norm()) / self.pixels_per_mm
            elapsed = frame_gap / self.fps
            speeds.append(distance_mm / elapsed)
            path_length_mm += distance_mm
            if angles[index] is not None and angles[index - 1] is not None:
                angular_velocities.append(
                    math.degrees(_angle_delta(angles[index], angles[index - 1])) / elapsed
                )
        net_distance_mm = float((centers[-1] - centers[0]).norm()) / self.pixels_per_mm
        regions = Counter(int(record.get('scene_region', -1)) for record in track_records)
        mean_angular = sum(angular_velocities) / max(len(angular_velocities), 1)
        oscillation = math.sqrt(sum(
            (value - mean_angular) ** 2 for value in angular_velocities
        ) / max(len(angular_velocities), 1))
        return {
            'track_id': int(track_records[0]['track_id']),
            'valid_frames': len(track_records),
            'mean_speed_mm_s': sum(speeds) / max(len(speeds), 1),
            'mean_angular_velocity_deg_s': mean_angular,
            'axis_oscillation_deg_s': oscillation,
            'tortuosity': path_length_mm / max(net_distance_mm, 1e-6),
            'dwell_seconds_by_region': {
                str(region): frames / self.fps for region, frames in regions.items() if region >= 0
            },
        }

    def _frame_interactions(self, frame_records, network):
        for left_index, left in enumerate(frame_records):
            left_center = self._center(left)
            for right in frame_records[left_index + 1:]:
                distance = float((left_center - self._center(right)).norm()) / self.pixels_per_mm
                if distance <= self.interaction_distance_mm:
                    edge = tuple(sorted((int(left['track_id']), int(right['track_id']))))
                    network[edge] += 1

    @staticmethod
    def _direction_entropy(angles, bins=8):
        if not angles:
            return 0.0
        counts = [0] * bins
        for angle in angles:
            counts[int(((angle + math.pi) / (2 * math.pi)) * bins) % bins] += 1
        entropy = 0.0
        for count in counts:
            if count:
                probability = count / len(angles)
                entropy -= probability * math.log(probability)
        return entropy / math.log(bins)

    def summarize(self):
        validate_result_schema(self.records)
        records = correct_head_tail_sequence(self.records)
        tracks = defaultdict(list)
        frames = defaultdict(list)
        for record in records:
            tracks[int(record['track_id'])].append(record)
            frames[int(record['frame_index'])].append(record)
        individuals = [
            result for result in (self._individual(records) for records in tracks.values())
            if result is not None
        ]

        heatmap = torch.zeros(self.heatmap_size, dtype=torch.float32)
        network = Counter()
        angles = []
        entrance_flux_count = 0
        previous_regions = {}
        nearest_neighbor_by_frame = []
        width, height = self.image_size
        for frame_index in sorted(frames):
            frame_records = frames[frame_index]
            centers = torch.stack([self._center(record) for record in frame_records]) if frame_records else None
            for record in frame_records:
                center = self._center(record)
                column = min(int(float(center[0]) / max(width, 1) * self.heatmap_size[1]), self.heatmap_size[1] - 1)
                row = min(int(float(center[1]) / max(height, 1) * self.heatmap_size[0]), self.heatmap_size[0] - 1)
                heatmap[max(row, 0), max(column, 0)] += 1
                angle = self._axis_angle(record)
                if angle is not None:
                    angles.append(angle)
                track_id = int(record['track_id'])
                region = int(record.get('scene_region', -1))
                if self.entrance_region is not None and region == self.entrance_region:
                    if previous_regions.get(track_id) not in (None, self.entrance_region):
                        entrance_flux_count += 1
                previous_regions[track_id] = region
            self._frame_interactions(frame_records, network)
            if centers is not None and len(centers) > 1:
                distances = torch.cdist(centers, centers)
                distances.fill_diagonal_(float('inf'))
                nearest_neighbor_by_frame.append(float(distances.min(dim=1).values.mean()) / self.pixels_per_mm)

        duration = (max(frames) - min(frames) + 1) / self.fps if frames else 0.0
        mean_speed = sum(item['mean_speed_mm_s'] for item in individuals) / max(len(individuals), 1)
        aggregation_change = 0.0
        if len(nearest_neighbor_by_frame) >= 2:
            aggregation_change = nearest_neighbor_by_frame[-1] - nearest_neighbor_by_frame[0]
        return {
            'schema_version': 1,
            'units': {
                'distance': 'mm', 'time': 's', 'speed': 'mm/s',
                'angular_velocity': 'deg/s', 'density_heatmap': 'detections/frame',
            },
            'rules': {
                'fps': self.fps,
                'pixels_per_mm': self.pixels_per_mm,
                'min_track_frames': self.min_track_frames,
                'missing_pose': 'excluded_from angular statistics',
                'missing_frame': 'elapsed time uses frame-index gap',
            },
            'individual': individuals,
            'group': {
                'entrance_flux_per_second': entrance_flux_count / max(duration, 1e-6),
                'density_heatmap': (heatmap / max(len(frames), 1)).tolist(),
                'interaction_network': [
                    {'source_track_id': edge[0], 'target_track_id': edge[1],
                     'interaction_seconds': count / self.fps}
                    for edge, count in sorted(network.items())
                ],
                'activity_index_mm_s': mean_speed,
                'direction_entropy': self._direction_entropy(angles),
                'aggregation_change_mm': aggregation_change,
            },
        }
