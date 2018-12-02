import unittest

import commons


class TestCommons(unittest.TestCase):
    def test_chunker(self):
        l = [0] * 10
        gen_chunks = commons.chunker(l, 5)
        chunks = list(gen_chunks)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 5)
        self.assertEqual(len(chunks[1]), 5)

        gen_chunks = commons.chunker(l, 1000)
        chunks = list(gen_chunks)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 10)

    def test_onehot(self):
        idxs = [0]
        onehot = commons.onehot(idxs)
        self.assertListEqual(onehot.tolist(), [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])

        idxs = [0, 5, 9]
        onehot = commons.onehot(idxs)
        self.assertListEqual(onehot.tolist(), [[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ]])

    def test_onehot_flatten(self):
        idxs = [0, 1]
        onehot = commons.onehot_flatten(idxs)
        self.assertEqual(onehot.tolist(), [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
