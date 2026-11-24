import sys
import unittest
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import refine_tracks, apply_masks, Cancelled, PlateFlow


def detection(x, kind='plate'):
    return dict(id=99, kind=kind, box=[x, 20, x+20, 30], score=.9, predicted=False)


def frames(times, positions):
    return [dict(time=t, boxes=[] if x is None else [detection(x)])
            for t, x in zip(times, positions)]


class MotionRefinementTests(unittest.TestCase):
    def test_image_motion_follows_turn_and_expires_from_real_detection(self):
        flow = PlateFlow()
        rng = np.random.default_rng(42)
        texture = rng.integers(0, 256, (24, 48, 3), dtype=np.uint8)
        data = []
        for t, x, y in [(0, 30, 30), (.04, 38, 30), (.08, 38, 38), (.25, 38, 38)]:
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[y:y+24, x:x+48] = texture
            observed = [dict(kind='plate', box=[x,y,x+48,y+24], score=.9)] if t == 0 else []
            boxes = flow.update(image, observed, t)
            data.append(dict(time=t, boxes=boxes))
            if t in (.04, .08):
                self.assertEqual(len(boxes), 1)
                np.testing.assert_allclose(boxes[0]['box'], [x,y,x+48,y+24], atol=1)
        refine_tracks(data, 160, 120)
        self.assertTrue(data[2]['boxes'][0]['predicted'])
        self.assertEqual(data[-1]['boxes'], [])

    def test_fast_motion_without_overlap_keeps_identity(self):
        data = frames([0, .04, .08, .12], [0, 15, 40, 65])
        refine_tracks(data, 200, 100)
        self.assertEqual({b['id'] for f in data for b in f['boxes']}, {1})
        self.assertEqual(data[-1]['boxes'][0]['box'], [65, 20, 85, 30])
        self.assertGreater(data[-1]['boxes'][0]['motion_margin'], 0)

    def test_vfr_gap_interpolation_replaces_stale_prediction(self):
        data = frames([0, .04, .10, .16], [0, 10, None, 40])
        data[2]['boxes'] = [dict(detection(10), predicted=True)]
        result = refine_tracks(data, 200, 100)
        self.assertEqual(result['interpolated_boxes'], 1)
        self.assertEqual(len(data[2]['boxes']), 1)
        box = data[2]['boxes'][0]
        self.assertAlmostEqual(box['box'][0], 25)
        self.assertEqual(box['correction'], 'interpolated')
        self.assertEqual(box['id'], data[-1]['boxes'][0]['id'])

    def test_long_gap_and_kind_change_are_not_joined(self):
        for change_kind in (False, True):
            data = frames([0, .1, .3 if not change_kind else .15], [0, None, 5])
            if change_kind:
                data[-1]['boxes'][0]['kind'] = 'head'
            result = refine_tracks(data, 200, 100)
            self.assertEqual(result['interpolated_boxes'], 0)
            self.assertNotEqual(data[0]['boxes'][0]['id'], data[-1]['boxes'][0]['id'])

    def test_ambiguous_targets_and_large_jump_are_not_joined(self):
        data = frames([0, .04], [0, 100])
        refine_tracks(data, 200, 100)
        self.assertNotEqual(data[0]['boxes'][0]['id'], data[1]['boxes'][0]['id'])
        data = frames([0, .04], [0, 5])
        data[0]['boxes'].append(detection(10))
        refine_tracks(data, 200, 100)
        initial = {b['id'] for b in data[0]['boxes']}
        observed = [b for b in data[1]['boxes'] if not b['predicted']]
        self.assertNotIn(observed[0]['id'], initial)

    def test_tail_prediction_moves_clips_and_expires(self):
        data = frames([0, .04, .08, .25], [70, 80, None, None])
        refine_tracks(data, 105, 100)
        self.assertEqual(data[2]['boxes'][0]['box'], [90, 20, 105., 30])
        self.assertEqual(data[-1]['boxes'], [])

    def test_extra_margin_applies_to_rendering_and_legacy_still_works(self):
        image = np.full((100, 100, 3), 200, np.uint8)
        box = detection(20)
        legacy = apply_masks(image, [box], 'solid', 0)
        expanded = apply_masks(image, [dict(box, motion_margin=.25)], 'solid', 0)
        self.assertTrue(np.all(legacy[22, 16] == 200))
        self.assertTrue(np.all(expanded[22, 16] == 24))
        self.assertTrue(np.all(expanded[0] == 200))

    def test_cancellation(self):
        with self.assertRaises(Cancelled):
            refine_tracks(frames([0], [0]), 200, 100, lambda: True)


if __name__ == '__main__':
    unittest.main()
