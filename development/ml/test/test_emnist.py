import pathlib
import unittest

from emnist_predictor import EmnistPredictor

SUPPORT_DIRNAME = pathlib.Path('__file__').parents[0].resolve() / 'support' / 'emnist'

class TestEmnistPredictor(unittest.TestCase):
    def test_filename(self):
        predictor = EmnistPredictor()
        for filename in SUPPORT_DIRNAME.glob('*.png'):
            pred, conf = predictor.predict(str(filename))
            print(f'Prediction: {pred} at confidence: {conf} for image with character {filename.stem}')
            self.assertEqual(pred, filename.stem)
