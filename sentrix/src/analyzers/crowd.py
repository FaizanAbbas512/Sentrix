"""
crowd.py
--------
Crowd analyzer — sabse simple. Frame mein kitne log hain count karo.
Threshold se zyada -> crowd alert (bheed / possible stampede risk).

Future scope (paper mein future work likh sakte ho): density map,
zone-wise counting, flow direction.
"""


class CrowdAnalyzer:
    def __init__(self, cfg):
        self.threshold = cfg["crowd"]["threshold"]

    def update(self, detections):
        """(count, is_crowded) return karta hai."""
        count = sum(1 for d in detections if d.is_person)
        return count, count >= self.threshold
