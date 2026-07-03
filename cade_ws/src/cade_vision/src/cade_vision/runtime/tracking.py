"""Tracking helpers used by person and gesture workers."""


class PersonIoUTracker:
    """Person-only IoU tracker aligned with the gesture RealSense tester."""

    def __init__(self, iou_threshold=0.25, max_missing_frames=15):
        self.iou_threshold = float(iou_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks = []
        self._next_id = 1
        self.missing_track_ids = []

    def assign(self, detections):
        self.missing_track_ids = []
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        matches = []
        candidates = []

        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                iou = self._iou(track.get("bbox"), detection.get("bbox"))
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_index, detection_index))

        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if (
                track_index not in unmatched_tracks
                or detection_index not in unmatched_detections
            ):
                continue
            matches.append((track_index, detection_index))
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        track_ids = [None] * len(detections)
        new_tracks = []
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            track_ids[detection_index] = track["id"]
            new_tracks.append(
                {
                    "id": track["id"],
                    "bbox": detections[detection_index].get("bbox"),
                    "missing": 0,
                }
            )

        for detection_index in sorted(unmatched_detections):
            track_id = self._next_id
            self._next_id += 1
            track_ids[detection_index] = track_id
            new_tracks.append(
                {
                    "id": track_id,
                    "bbox": detections[detection_index].get("bbox"),
                    "missing": 0,
                }
            )

        for track_index in sorted(unmatched_tracks):
            track = dict(self._tracks[track_index])
            track["missing"] = int(track.get("missing", 0)) + 1
            self.missing_track_ids.append(track["id"])
            if track["missing"] <= self.max_missing_frames:
                new_tracks.append(track)

        self._tracks = new_tracks
        return track_ids

    def clear(self):
        self._tracks.clear()
        self._next_id = 1
        self.missing_track_ids = []

    @staticmethod
    def _iou(box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

