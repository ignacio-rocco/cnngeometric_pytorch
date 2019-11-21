import torch
from torchvision import transforms
from torch.autograd import Variable


class NormalizeImageDict(object):
    """
    
    Normalizes Tensor images in dictionary
    
    Args:
        image_keys (list): dict. keys of the images to be normalized
        normalizeRange (bool): if True the image is divided by 255.0s
    
    """

    def __init__(self, image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0                
            sample[key] = self.normalize(sample[key])
        return sample

    
def normalize_image(image, forward=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)):

        mean = list(mean)
        std = list(std)

        im_size = image.size()
        mean = torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
        std = torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
        if image.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        if isinstance(image, torch.autograd.Variable):
            mean = Variable(mean, requires_grad=False)
            std = Variable(std, requires_grad=False)
        if forward:
            if len(im_size) == 3:
                result = image.sub(mean.expand(im_size)).div(std.expand(im_size))
            elif len(im_size) == 4:
                result = image.sub(mean.unsqueeze(0).expand(im_size)).div(std.unsqueeze(0).expand(im_size))
            else:
                raise TypeError("Couldn't read image due to an unexpected format")

        else:
            if len(im_size) == 3:
                result = image.mul(std.expand(im_size)).add(mean.expand(im_size))
            elif len(im_size) == 4:
                result = image.mul(std.unsqueeze(0).expand(im_size)).add(mean.unsqueeze(0).expand(im_size))
            else:
                raise TypeError("Couldn't read image due to an unexpected format")

        return result
