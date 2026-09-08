"""Fixed-parameter ByteTrack + OC-SORT style pose-motion tracker."""

import math
from dataclasses import dataclass

import torch
import torchvision


def _linear_sum_assignment(cost):
    """Rectangular Hungarian assignment without a SciPy runtime dependency."""
    if cost.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=cost.device)
        return empty, empty
    original_device = cost.device
    matrix = cost.detach().float().cpu()
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.t()
    rows, columns = matrix.shape
    finite = matrix[torch.isfinite(matrix)]
    large = float(finite.max().item() + 1e6) if finite.numel() else 1e9
    values = matrix.nan_to_num(nan=large, posinf=large, neginf=large).tolist()
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minimum = [float('inf')] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float('inf')
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = values[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    row_indices = []
    column_indices = []
    for column in range(1, columns + 1):
        if p[column]:
            row_indices.append(p[column] - 1)
            column_indices.append(column - 1)
    row_indices = torch.tensor(row_indices, dtype=torch.long, device=original_device)
    column_indices = torch.tensor(column_indices, dtype=torch.long, device=original_device)
    if transposed:
        return column_indices, row_indices
    return row_indices, column_indices


def _box_measurement(box):
    x1, y1, x2, y2 = box
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))


def _measurement_box(measurement):
    cx, cy, width, height = measurement
    return torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))


@dataclass(frozen=True)
class TrackerConfig:
    high_score_threshold: float = 0.50
    low_score_threshold: float = 0.10
    new_track_threshold: float = 0.60
    max_missed: int = 30
    min_hits: int = 2
    mahalanobis_gate: float = 13.28
    max_assignment_cost: float = 3.0
    process_noise: float = 1.0
    measurement_noise: float = 4.0
    observation_velocity_blend: float = 0.8
    swap_transition_penalty: float = 0.05
    weight_mahalanobis: float = 0.35
    weight_iou: float = 0.35
    weight_endpoint_oks: float = 0.10
    weight_axis: float = 0.08
    weight_length_ratio: float = 0.07
    weight_region: float = 0.05

    @classmethod
    def from_calibration_contract(cls, contract):
        tracker = contract.get('tracker', contract)
        if tracker.get('source_data') != 'calibration':
            raise ValueError('Tracker parameters must be fitted on calibration.')
        if tracker.get('effective_stage') != 'E-S5':
            raise ValueError('Tracker parameters must be frozen at E-S5.')
        parameters = tracker.get('parameters', {})
        required = set(cls.__dataclass_fields__)
        missing = required - set(parameters)
        extra = set(parameters) - required
        if missing or extra:
            raise ValueError(
                f'Tracker contract fields mismatch: missing={sorted(missing)}, '
                f'extra={sorted(extra)}'
            )
        return cls(**parameters)


class _Track:
    def __init__(self, track_id, detection, frame_index, config):
        measurement = _box_measurement(detection['box'])
        self.track_id = track_id
        self.state = measurement.new_tensor([
            measurement[0], measurement[1], 0.0, 0.0, measurement[2], measurement[3]
        ])
        self.covariance = torch.eye(6, device=measurement.device, dtype=measurement.dtype) * 10.0
        self.score = float(detection['score'])
        self.quality = float(detection.get('quality', detection['score']))
        self.region = int(detection.get('region', -1))
        self.query_index = int(detection.get('query_index', -1))
        self.age = 1
        self.hits = 1
        self.missed = 0
        self.last_frame = frame_index
        self.last_observation_frame = frame_index
        self.last_observation_center = measurement[:2].clone()
        keypoints = detection.get('keypoints')
        self.raw_keypoints = keypoints.clone() if keypoints is not None else None
        self.keypoint_states = None
        self.orientation_cost = measurement.new_zeros(2)
        self.keypoints = None
        self._update_orientation(keypoints, config)

    @property
    def box(self):
        return _measurement_box(self.state[[0, 1, 4, 5]])

    def predict(self, config):
        transition = torch.eye(6, device=self.state.device, dtype=self.state.dtype)
        transition[0, 2] = 1.0
        transition[1, 3] = 1.0
        process = torch.eye(6, device=self.state.device, dtype=self.state.dtype) * config.process_noise
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.t() + process
        self.age += 1

    def innovation(self, box, config):
        measurement = _box_measurement(box)
        observation = torch.zeros(4, 6, device=box.device, dtype=box.dtype)
        observation[0, 0] = observation[1, 1] = 1.0
        observation[2, 4] = observation[3, 5] = 1.0
        covariance = observation @ self.covariance @ observation.t()
        covariance = covariance + torch.eye(4, device=box.device, dtype=box.dtype) * config.measurement_noise
        residual = measurement - observation @ self.state
        distance = residual @ torch.linalg.solve(covariance, residual)
        return residual, covariance, observation, distance

    def update(self, detection, frame_index, config):
        residual, covariance, observation, _ = self.innovation(detection['box'], config)
        gain = self.covariance @ observation.t() @ torch.linalg.inv(covariance)
        self.state = self.state + gain @ residual
        identity = torch.eye(6, device=self.state.device, dtype=self.state.dtype)
        self.covariance = (identity - gain @ observation) @ self.covariance

        center = _box_measurement(detection['box'])[:2]
        frame_gap = max(frame_index - self.last_observation_frame, 1)
        observed_velocity = (center - self.last_observation_center) / frame_gap
        blend = config.observation_velocity_blend
        self.state[2:4] = blend * observed_velocity + (1.0 - blend) * self.state[2:4]
        self.last_observation_center = center.clone()
        self.last_observation_frame = frame_index
        self.last_frame = frame_index
        self.score = float(detection['score'])
        self.quality = float(detection.get('quality', detection['score']))
        self.region = int(detection.get('region', self.region))
        self.query_index = int(detection.get('query_index', -1))
        self.hits += 1
        self.missed = 0
        self._update_orientation(detection.get('keypoints'), config)

    def mark_missed(self):
        self.missed += 1

    def _update_orientation(self, keypoints, config):
        if keypoints is None:
            return
        states = torch.stack((keypoints, keypoints.flip(0)))
        if self.keypoint_states is None:
            self.keypoint_states = states
            self.orientation_cost = states.new_tensor([
                0.0, 2.0 * config.swap_transition_penalty
            ])
        else:
            transition_cost = torch.cdist(
                self.keypoint_states.reshape(2, -1), states.reshape(2, -1), p=2
            )
            scale = torch.linalg.vector_norm(self.state[4:6]).clamp_min(1.0)
            transition_cost = transition_cost / scale
            transition_cost += states.new_tensor([
                [0.0, config.swap_transition_penalty],
                [config.swap_transition_penalty, 0.0],
            ])
            accumulated = self.orientation_cost[:, None] + transition_cost
            self.orientation_cost = accumulated.min(dim=0).values
            self.keypoint_states = states
        selected = int(self.orientation_cost.argmin())
        self.raw_keypoints = keypoints.clone()
        self.keypoints = states[selected].clone()


class PoseMotionTracker:
    """Track-by-detection without ReID or detector query embeddings."""

    def __init__(self, config=None):
        self.config = config or TrackerConfig()
        self.tracks = []
        self.next_track_id = 1

    @staticmethod
    def _detections(detections):
        boxes = detections['boxes']
        scores = detections['scores']
        keypoints = detections.get('keypoints')
        qualities = detections.get('quality', scores)
        regions = detections.get('scene_region')
        queries = detections.get('query_indices')
        result = []
        for index in range(len(boxes)):
            result.append({
                'box': boxes[index],
                'score': scores[index],
                'quality': qualities[index],
                'keypoints': keypoints[index] if keypoints is not None else None,
                'region': regions[index] if regions is not None else -1,
                'query_index': queries[index] if queries is not None else -1,
            })
        return result

    @staticmethod
    def _endpoint_oks(track, detection):
        if track.keypoints is None or detection.get('keypoints') is None:
            return track.state.new_tensor(0.5)
        diagonal = torch.linalg.vector_norm(track.state[4:6]).clamp_min(1.0)
        squared = (track.keypoints - detection['keypoints']).square().sum(-1)
        return torch.exp(-squared / (2.0 * (0.10 * diagonal) ** 2)).mean()

    def _cost(self, tracks, detections):
        if not tracks or not detections:
            device = tracks[0].state.device if tracks else detections[0]['box'].device
            return torch.zeros((len(tracks), len(detections)), device=device)
        track_boxes = torch.stack([track.box for track in tracks])
        detection_boxes = torch.stack([detection['box'] for detection in detections])
        iou = torchvision.ops.box_iou(track_boxes, detection_boxes)
        cost = iou.new_full(iou.shape, float('inf'))
        cfg = self.config
        for row, track in enumerate(tracks):
            for column, detection in enumerate(detections):
                _, _, _, mahalanobis = track.innovation(detection['box'], cfg)
                if float(mahalanobis) > cfg.mahalanobis_gate:
                    continue
                endpoint_cost = 1.0 - self._endpoint_oks(track, detection)
                axis_cost = iou.new_tensor(0.5)
                length_cost = iou.new_tensor(0.0)
                if track.keypoints is not None and detection.get('keypoints') is not None:
                    track_axis = track.keypoints[1] - track.keypoints[0]
                    detection_axis = detection['keypoints'][1] - detection['keypoints'][0]
                    track_length = track_axis.norm().clamp_min(1e-6)
                    detection_length = detection_axis.norm().clamp_min(1e-6)
                    cosine = torch.dot(track_axis, detection_axis).abs() / (track_length * detection_length)
                    axis_cost = 1.0 - cosine.clamp(0, 1)
                    length_cost = torch.log(detection_length / track_length).abs().clamp_max(3.0) / 3.0
                region_cost = float(
                    track.region >= 0 and int(detection.get('region', -1)) >= 0
                    and track.region != int(detection['region'])
                )
                cost[row, column] = (
                    cfg.weight_mahalanobis * (mahalanobis / cfg.mahalanobis_gate)
                    + cfg.weight_iou * (1.0 - iou[row, column])
                    + cfg.weight_endpoint_oks * endpoint_cost
                    + cfg.weight_axis * axis_cost
                    + cfg.weight_length_ratio * length_cost
                    + cfg.weight_region * region_cost
                )
        return cost

    def _associate(self, track_indices, detections, detection_indices):
        if not track_indices or not detection_indices:
            return [], set(), set()
        tracks = [self.tracks[index] for index in track_indices]
        selected = [detections[index] for index in detection_indices]
        cost = self._cost(tracks, selected)
        rows, columns = _linear_sum_assignment(cost)
        matches = []
        used_tracks = set()
        used_detections = set()
        for row, column in zip(rows.tolist(), columns.tolist()):
            if not torch.isfinite(cost[row, column]) or float(cost[row, column]) > self.config.max_assignment_cost:
                continue
            track_index = track_indices[row]
            detection_index = detection_indices[column]
            matches.append((track_index, detection_index))
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        return matches, used_tracks, used_detections

    def update(self, detections, frame_index):
        detections = self._detections(detections)
        for track in self.tracks:
            track.predict(self.config)

        high = [index for index, detection in enumerate(detections)
                if float(detection['score']) >= self.config.high_score_threshold]
        low = [index for index, detection in enumerate(detections)
               if self.config.low_score_threshold <= float(detection['score']) < self.config.high_score_threshold]
        active = list(range(len(self.tracks)))
        first, used_tracks, used_high = self._associate(active, detections, high)
        remaining_tracks = [index for index in active if index not in used_tracks]
        second, used_tracks_low, _ = self._associate(remaining_tracks, detections, low)
        matches = first + second
        all_used_tracks = used_tracks | used_tracks_low
        for track_index, detection_index in matches:
            self.tracks[track_index].update(detections[detection_index], frame_index, self.config)
        for track_index, track in enumerate(self.tracks):
            if track_index not in all_used_tracks:
                track.mark_missed()

        for detection_index in high:
            if detection_index in used_high:
                continue
            detection = detections[detection_index]
            if float(detection['score']) < self.config.new_track_threshold:
                continue
            self.tracks.append(_Track(
                self.next_track_id, detection, frame_index, self.config
            ))
            self.next_track_id += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.config.max_missed]

        output = []
        for track in self.tracks:
            if track.missed or track.hits < self.config.min_hits:
                continue
            output.append({
                'schema_version': 1,
                'frame_index': int(frame_index),
                'track_id': track.track_id,
                'query_index': track.query_index,
                'box': track.box.detach().clone(),
                'keypoints': track.keypoints.detach().clone() if track.keypoints is not None else None,
                'score': track.score,
                'quality': track.quality,
                'scene_region': track.region,
                'track_age': track.age,
                'track_hits': track.hits,
            })
        return output


def correct_head_tail_sequence(records, swap_transition_penalty=0.05):
    """用完整轨迹Viterbi回溯一次性修正头尾方向，而非逐帧贪心翻转。"""
    corrected = [dict(record) for record in records]
    by_track = {}
    for index, record in enumerate(corrected):
        if record.get('keypoints') is not None:
            by_track.setdefault(int(record['track_id']), []).append(index)
    for indices in by_track.values():
        indices.sort(key=lambda index: int(corrected[index]['frame_index']))
        if len(indices) < 2:
            continue
        states = [torch.stack((
            corrected[index]['keypoints'], corrected[index]['keypoints'].flip(0)
        )) for index in indices]
        costs = states[0].new_tensor([0.0, float(swap_transition_penalty)])
        backpointers = []
        for position in range(1, len(states)):
            previous = states[position - 1].reshape(2, -1)
            current = states[position].reshape(2, -1)
            box = corrected[indices[position]]['box'].float()
            scale = torch.linalg.vector_norm(box[2:] - box[:2]).clamp_min(1.0)
            transition = torch.cdist(previous, current, p=2) / scale
            transition = transition + transition.new_tensor([
                [0.0, float(swap_transition_penalty)],
                [float(swap_transition_penalty), 0.0],
            ])
            candidates = costs[:, None] + transition
            costs, predecessor = candidates.min(dim=0)
            backpointers.append(predecessor)
        selected = [int(costs.argmin())]
        for predecessor in reversed(backpointers):
            selected.append(int(predecessor[selected[-1]]))
        selected.reverse()
        for index, state_index, state in zip(indices, selected, states):
            corrected[index]['keypoints'] = state[state_index].clone()
    return corrected
