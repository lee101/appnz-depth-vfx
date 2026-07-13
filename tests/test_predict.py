import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import contract as MODULE


class ContractTest(unittest.TestCase):
    def test_effect_extensions_are_explicit(self):
        self.assertEqual(MODULE.output_suffix("parallax"), ".mp4")
        self.assertEqual(MODULE.output_suffix("normals"), ".png")

    def test_unknown_effect_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.output_suffix("surprise")

    def test_dimensions_are_bounded(self):
        MODULE.validate_dimensions(1920, 1080)
        with self.assertRaises(ValueError):
            MODULE.validate_dimensions(10000, 10000)


if __name__ == "__main__":
    unittest.main()
