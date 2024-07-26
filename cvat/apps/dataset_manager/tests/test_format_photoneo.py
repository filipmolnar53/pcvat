import unittest
import tempfile
import os
import json
# from unittest.mock import patch, MagicMock
# from datumaro.components.dataset import Dataset
# from datumaro.components.annotation import AnnotationType, LabelCategories
# from cvat.apps.dataset_manager.bindings import CVATDataExtractor
# from datumaro.plugins.coco_format.importer import CocoImporter
from cvat.apps.dataset_manager.formats.photoneo import *

# Mock the functions from the cvat.apps.dataset_manager module
# from cvat.apps.dataset_manager.bindings import detect_dataset, import_dm_annotations

class TestCOCOPhotoneoImporter(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(3,3)

if __name__ == '__main__':
    unittest.main()
