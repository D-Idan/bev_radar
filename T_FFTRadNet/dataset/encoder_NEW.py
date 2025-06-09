import numpy as np

class RAEncoder:
    def __init__(self, geometry, statistics, regression_layer=2, verbose=False):
        """
        Initialize the RAEncoder with geometry and statistics.

        Parameters:
            geometry (dict): Contains 'resolution', 'ranges', and optionally 'size'.
            statistics (dict): Contains 'reg_mean' and 'reg_std' for regression normalization.
            regression_layer (int): Number of regression layers (default is 2).
            verbose (bool): If True, print debug statements.
        """
        assert 'resolution' in geometry and len(geometry['resolution']) >= 2
        assert 'ranges' in geometry and len(geometry['ranges']) >= 2
        assert 'reg_mean' in statistics and 'reg_std' in statistics

        self.geometry = geometry
        self.statistics = statistics
        self.regression_layer = regression_layer
        self.verbose = verbose

        self.INPUT_DIM = tuple(geometry['ranges'])
        self.OUTPUT_DIM = (
            regression_layer + 1,
            self.INPUT_DIM[0] // 4,
            self.INPUT_DIM[1] // 4
        )

    def _get_bins(self, lab):
        scale = 4
        r_res, a_res = self.geometry['resolution'][:2]
        range_bin = int(np.clip(lab[0] / r_res / scale, 0, self.OUTPUT_DIM[1] - 1))
        angle_bin = int(np.clip(np.floor(lab[1] / a_res / scale + self.OUTPUT_DIM[2] / 2), 0, self.OUTPUT_DIM[2] - 1))
        return range_bin, angle_bin

    def _get_local_grid(self, size, resolution, scale):
        s = int((size - 1) / 2)
        axis = np.linspace(resolution * s, -resolution * s, size) * scale
        return np.meshgrid(axis, axis)

    def encode(self, labels):
        """
        Encode labels into a RA map tensor with classification and regression information.

        Parameters:
            labels (list of lists): Each label is [range, angle].

        Returns:
            np.ndarray: Encoded 3D map with shape (channels, range_bins, angle_bins).
        """
        encoded_map = np.zeros(self.OUTPUT_DIM)
        scale = 4

        for lab in labels:
            if lab[0] < 0:
                continue

            r_bin, a_bin = self._get_bins(lab)
            r_res, a_res = self.geometry['resolution'][:2]
            r_mod = lab[0] - r_bin * r_res * scale
            a_mod = lab[1] - (a_bin - self.OUTPUT_DIM[2] / 2) * a_res * scale

            if self.geometry.get('size', 1) == 1:
                encoded_map[0, r_bin, a_bin] = 1
                encoded_map[1, r_bin, a_bin] = (r_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                encoded_map[2, r_bin, a_bin] = (a_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
            else:
                size = self.geometry['size']
                px_a, px_r = self._get_local_grid(size, r_res, scale)
                center_r = r_bin - size // 2
                center_a = a_bin - size // 2

                for r_off in range(size):
                    for a_off in range(size):
                        rr = center_r + r_off
                        aa = center_a + a_off

                        if 0 <= rr < self.OUTPUT_DIM[1] and 0 <= aa < self.OUTPUT_DIM[2]:
                            encoded_map[0, rr, aa] = 1
                            encoded_map[1, rr, aa] = (r_mod - px_r[r_off, a_off] - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                            encoded_map[2, rr, aa] = (a_mod - px_a[r_off, a_off] - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]

        return encoded_map

    def decode(self, encoded_map, threshold):
        """
        Decode an encoded RA map back into labeled coordinates.

        Parameters:
            encoded_map (np.ndarray): The encoded map.
            threshold (float): Threshold for detecting valid objects.

        Returns:
            list: Decoded objects as [range, angle, confidence].
        """
        coords = []
        r_bins, a_bins = np.where(encoded_map[0] >= threshold)

        for r_bin, a_bin in zip(r_bins, a_bins):
            R = r_bin * 4 * self.geometry['resolution'][0] + \
                encoded_map[1, r_bin, a_bin] * self.statistics['reg_std'][0] + self.statistics['reg_mean'][0]

            A = (a_bin - self.OUTPUT_DIM[2] / 2) * 4 * self.geometry['resolution'][1] + \
                encoded_map[2, r_bin, a_bin] * self.statistics['reg_std'][1] + self.statistics['reg_mean'][1]

            C = encoded_map[0, r_bin, a_bin]
            coords.append([R, A, C])

        return coords
