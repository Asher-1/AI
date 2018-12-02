import unittest

from boundingbox import BoundingBox
from configs import ADNetConf


class TestCommons(unittest.TestCase):
    def test_iou(self):
        a = BoundingBox(253, 66, 30, 30)
        b = BoundingBox(251, 67, 32, 32)
        iou = a.iou(b)
        iou2 = b.iou(a)
        self.assertEqual(iou, iou2)
        self.assertGreater(iou, 0.8)

    def test_do_action(self):
        ADNetConf.conf = ADNetConf(None)
        ADNetConf.conf.conf = {
            'action_move': {
                'x': 0.03, 'y': 0.03, 'w': 0.03, 'h': 0.03
            },
            'predict': {
                'stop_iou': 0.93
            }
        }

        a = BoundingBox(253, 66, 30, 30)
        b = BoundingBox(251, 67, 32, 32)
        lb = BoundingBox.get_action_label(b, a)
        self.assertIn(lb, [4, 5])

        # default do_action test
        a = BoundingBox(252, 65, 25, 30)

        b = a.do_action(None, 0)
        c = BoundingBox(251, 65, 25, 30)
        self.assertEqual(b, c)
        b = a.do_action(None, 1)
        c = BoundingBox(250, 65, 25, 30)
        self.assertEqual(b, c)

        b = a.do_action(None, 2)
        c = BoundingBox(253, 65, 25, 30)
        self.assertEqual(b, c)
        b = a.do_action(None, 3)
        c = BoundingBox(254, 65, 25, 30)
        self.assertEqual(b, c)

        b = a.do_action(None, 4)
        c = BoundingBox(252, 64, 25, 30)
        self.assertEqual(b, c)
        b = a.do_action(None, 5)
        c = BoundingBox(252, 63, 25, 30)
        self.assertEqual(b, c)

        b = a.do_action(None, 6)
        c = BoundingBox(252, 66, 25, 30)
        self.assertEqual(b, c)
        b = a.do_action(None, 7)
        c = BoundingBox(252, 67, 25, 30)
        self.assertEqual(b, c)

        b = a.do_action(None, 8)
        c = BoundingBox(252, 65, 25, 30)
        self.assertEqual(b, c)

        b = a.do_action(None, 9)
        c = BoundingBox(253, 66, 23, 28)
        self.assertEqual(b, c)
        b = a.do_action(None, 10)
        c = BoundingBox(251, 64, 27, 32)
        self.assertEqual(b, c)

        # box not moved example
        a = BoundingBox(252, 65, 25, 30)
        b = a.do_action(None, 2)
        self.assertNotEqual(a, b)
