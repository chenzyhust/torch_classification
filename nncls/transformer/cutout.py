import torch
import numpy

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, prob):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if numpy.random.random() < self.prob:
            return img
        h = img.size(1)
        w = img.size(2)

        mask = numpy.ones((h, w), numpy.float32)

        for n in range(self.n_holes):
            y = numpy.random.randint(h)
            x = numpy.random.randint(w)

            y1 = numpy.clip(y - self.length // 2, 0, h)
            y2 = numpy.clip(y + self.length // 2, 0, h)
            x1 = numpy.clip(x - self.length // 2, 0, w)
            x2 = numpy.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class DualCutout:
    def __init__(self, n_holes, length, prob):
        self.cutout = Cutout(n_holes, length, prob)

    def __call__(self, image) :
        return np.hstack([self.cutout(image), self.cutout(image)])