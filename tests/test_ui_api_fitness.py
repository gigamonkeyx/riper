#!/usr/bin/env python3
"""
UI API fitness_score field validation
Ensures that the /api/simulation/status endpoint does not report out-of-range values.
"""

import unittest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ui_api import app, initialize_simulation


class TestUIAPIFitnessField(unittest.TestCase):
    def setUp(self):
        initialize_simulation()
        self.client = app.test_client()

    def test_status_endpoint_has_no_invalid_fitness_field(self):
        resp = self.client.get('/api/simulation/status')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        # fitness_score should not be present or must be within [0.0, 1.0]
        if 'fitness_score' in data:
            self.assertIsInstance(data['fitness_score'], (int, float))
            self.assertGreaterEqual(data['fitness_score'], 0.0)
            self.assertLessEqual(data['fitness_score'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

