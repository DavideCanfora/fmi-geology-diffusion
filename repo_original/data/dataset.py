import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """
    Build a list of image paths from either a directory or a file list.

    If `dir` is a file, it is interpreted as a text file containing image paths.
    If `dir` is a directory, all image files inside it are collected
    recursively and sorted.

    For the FMI experiments, `data_root` is usually a directory containing PNG
    patches extracted from DLIS files.
    """
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    """
    Generic inpainting dataset used by the original Palette repository.

    It loads RGB images, resizes and normalizes them to [-1, 1], generates an
    artificial mask, and returns the ground-truth image together with a
    conditional image where the masked region is replaced by Gaussian noise.

    Returned fields:
        gt_image   : original image used as reconstruction target
        cond_image : conditional input, with masked pixels replaced by noise
        mask_image : visualization image, with masked pixels set to white
        mask       : binary inpainting mask, where 1 indicates the hole
        path       : image filename
    """
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)




class FMIInpaintDataset(InpaintDataset):
    """
    FMI-aware inpainting dataset.

    FMI borehole images often contain real missing regions, visible as black
    vertical bands caused by incomplete pad coverage or invalid measurements.
    These real gaps should not be treated as artificial training targets.

    This dataset extends the generic InpaintDataset by:
    1. detecting the valid FMI support region from non-black pixels;
    2. generating a standard artificial inpainting mask;
    3. intersecting the artificial mask with the valid support region;
    4. producing a conditional image only where valid pixels were deliberately
       hidden.

    In this way, the model is trained to reconstruct deliberately removed FMI
    information, not regions where the original acquisition already contains no
    signal.

    Additional returned field:
        valid_region : binary map of non-black FMI support pixels.
    """

    def __init__(
        self,
        data_root,
        mask_config={},
        data_len=-1,
        image_size=[256, 256],
        black_threshold=-0.95,
        loader=pil_loader
    ):
        super().__init__(
            data_root=data_root,
            mask_config=mask_config,
            data_len=data_len,
            image_size=image_size,
            loader=loader
        )
        self.black_threshold = black_threshold

    def get_valid_region(self, img):
        """
        Estimate the valid support of an FMI patch.

        Images are normalized to [-1, 1]. Missing FMI areas are rendered as
        black pixels, approximately equal to -1 in all channels. A pixel is
        considered valid if at least one RGB channel is above black_threshold.

        Returns:
            Tensor of shape [1, H, W], with 1 for valid FMI pixels and 0 for
            already-missing regions.
        """
        # img is normalized in [-1, 1], shape [3, H, W]
        # black FMI gaps are close to -1 in all RGB channels.
        # valid_region shape: [1, H, W], values 0/1.
        valid = (img > self.black_threshold).any(dim=0, keepdim=True).float()
        return valid

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]

        img = self.tfs(self.loader(path))

        raw_mask = self.get_mask().float()
        valid_region = self.get_valid_region(img)

        # Restrict artificial holes to valid FMI pixels. Real black bands
        # remain visible as missing acquisition regions and are not used as
        # artificial reconstruction targets.
        # final mask: only mask valid FMI pixels, never already-black gaps
        mask = raw_mask * valid_region

        # if the mask accidentally becomes almost empty, retry a few times
        tries = 0
        while mask.mean() < 0.01 and tries < 10:
            raw_mask = self.get_mask().float()
            mask = raw_mask * valid_region
            tries += 1

        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['valid_region'] = valid_region
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret



class FMIRealGapDataset(FMIInpaintDataset):
    """
    FMI dataset for real-gap inference.

    This class identifies real missing FMI bands from the input image itself.

    Real FMI gaps are treated as vertically structured black bands, not as
    arbitrary isolated black pixels. The mask combines:
    1. large vertical blank strips;
    2. thinner vertical black strips.

    Convention used by Palette:
        mask = 1 means hole / region to generate
        mask = 0 means observed context / region to preserve

    The original incomplete FMI image is used as gt_image only because the
    current Palette restoration API expects y_0 for clamping observed pixels.
    It is not a true complete ground truth.
    """

    def __init__(
        self,
        data_root,
        mask_config={},
        data_len=-1,
        image_size=[256, 256],
        black_threshold=-0.95,
        column_black_ratio=0.90,
        thin_column_black_ratio=0.35,
        horizontal_dilation=1,
        loader=pil_loader
    ):
        super().__init__(
            data_root=data_root,
            mask_config=mask_config,
            data_len=data_len,
            image_size=image_size,
            black_threshold=black_threshold,
            loader=loader
        )
        self.column_black_ratio = column_black_ratio
        self.thin_column_black_ratio = thin_column_black_ratio
        self.horizontal_dilation = horizontal_dilation

    def get_black_pixel_map(self, img):
        """
        Detect nearly black pixels.

        img is normalized in [-1, 1], shape [3, H, W].
        A pixel is black only if all RGB channels are below black_threshold.
        """
        return (img <= self.black_threshold).all(dim=0, keepdim=True).float()

    def horizontal_dilate_mask(self, mask):
        """
        Slightly expand vertical gaps horizontally.

        This helps include very thin black bands and their immediate borders
        without turning isolated dark geological pixels into large holes.
        """
        if self.horizontal_dilation <= 0:
            return mask

        kernel_size = 2 * self.horizontal_dilation + 1
        mask_bchw = mask.unsqueeze(0)
        dilated = torch.nn.functional.max_pool2d(
            mask_bchw,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, self.horizontal_dilation)
        )
        return dilated.squeeze(0)

    def get_real_gap_mask(self, img):
        """
        Detect real FMI gaps as vertically structured black bands.

        Large gaps are columns that are almost fully black.
        Thin gaps are columns that have a meaningful vertical concentration of
        black pixels, then are slightly dilated horizontally.
        """
        black_pixels = self.get_black_pixel_map(img)

        # black_pixels: [1, H, W]
        # column_black_fraction: [1, 1, W]
        column_black_fraction = black_pixels.mean(dim=1, keepdim=True)

        large_gap_columns = (column_black_fraction >= self.column_black_ratio).float()
        thin_gap_columns = (column_black_fraction >= self.thin_column_black_ratio).float()

        large_gap_mask = black_pixels * large_gap_columns
        thin_gap_mask = black_pixels * thin_gap_columns

        real_gap_mask = torch.maximum(large_gap_mask, thin_gap_mask)
        real_gap_mask = self.horizontal_dilate_mask(real_gap_mask)
        real_gap_mask = real_gap_mask.clamp(0.0, 1.0)

        return real_gap_mask

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]

        img = self.tfs(self.loader(path))

        real_gap_mask = self.get_real_gap_mask(img)
        valid_region = 1.0 - real_gap_mask

        cond_image = img * (1.0 - real_gap_mask) + real_gap_mask * torch.randn_like(img)
        mask_img = img * (1.0 - real_gap_mask) + real_gap_mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = real_gap_mask
        ret['valid_region'] = valid_region
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret
