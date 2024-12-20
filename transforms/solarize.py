# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import numpy as np
from PIL import ImageOps
from PIL.Image import Image as PILImage


class RandomSolarization(object):

    def __init__(self, prob: float = 0.5, threshold: int = 128):
        self.prob = prob
        self.threshold = threshold

    def __call__(self, sample: PILImage) -> PILImage:
        """Solarizes the given input image

        Args:
            sample:
                PIL image to which solarize will be applied.

        Returns:
            Solarized image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            # return solarized image
            return ImageOps.solarize(sample, threshold=self.threshold)
        # return original image
        return sample