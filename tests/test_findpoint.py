import unittest

import numpy as np

from post_wrf.findpoint import nearest_position, nearest_positions, weighted_average


class FindPointTestCase(unittest.TestCase):
    def setUp(self):
        self.lon2d, self.lat2d = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        self.data2d = np.arange(121.0).reshape(11, 11)

    def test_nearest_positions_respects_num_and_order(self):
        indices = nearest_positions(3.3, 7.8, self.lon2d, self.lat2d, num=4)
        self.assertEqual(len(indices[0]), 4)

        distances = (
            (3.3 - self.lon2d[indices]) ** 2
            + (7.8 - self.lat2d[indices]) ** 2
        )
        self.assertTrue(np.all(distances[:-1] <= distances[1:]))

    def test_weighted_average_handles_nan_neighbors(self):
        i_indices, j_indices = nearest_positions(3.3, 7.8, self.lon2d, self.lat2d, num=4)
        data = self.data2d.copy()
        data[i_indices[0], j_indices[0]] = np.nan

        value = weighted_average(3.3, 7.8, self.lon2d, self.lat2d, data, num=4)
        self.assertTrue(np.isfinite(value))

    def test_weighted_average_returns_nan_when_all_neighbors_invalid(self):
        i_indices, j_indices = nearest_positions(3.3, 7.8, self.lon2d, self.lat2d, num=4)
        data = self.data2d.copy()
        data[i_indices, j_indices] = np.nan

        value = weighted_average(3.3, 7.8, self.lon2d, self.lat2d, data, num=4)
        self.assertTrue(np.isnan(value))

    def test_weighted_average_exact_grid_hit_returns_point_value(self):
        value = weighted_average(5.0, 5.0, self.lon2d, self.lat2d, self.data2d, num=4)
        self.assertEqual(value, self.data2d[5, 5])

    def test_nearest_position_exact_grid_hit(self):
        self.assertEqual(nearest_position(5.0, 5.0, self.lon2d, self.lat2d), (5, 5))


if __name__ == "__main__":
    unittest.main()
