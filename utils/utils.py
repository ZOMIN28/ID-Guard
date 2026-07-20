import cv2
import torch
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torchvision.transforms import Resize
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.utils as vutils


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    if x.min() < 0:
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    else:
        return x


# process celebA labels
def create_labels(c_org, c_dim=5, selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.clone()
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch


def imFromAttReg(att, reg, x_real):
        """Mixes attention, color and real images"""
        return (1-att)*reg + att*x_real


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
    

def resize_image(image, sizes):
    return F.interpolate(image, size=(sizes,sizes), mode='bilinear', align_corners=True)


def Image2tensor(imagepath, process=False, resize=256):
    img = Image.open(imagepath).convert("RGB")
    transform = []
    transform.append(T.ToTensor())
    if len(img.split()) == 3:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        #transform.append(T.GaussianBlur(kernel_size=5, sigma=(2, 2)))
    else:
       transform.append(T.Normalize(mean=0.5, std=0.5))
    if process:
        transform.append(T.Resize([resize,resize]))

    transform = T.Compose(transform)
    img = torch.unsqueeze(transform(img), dim=0).to(device)
    return img


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def save_heatmap_overlay(x_real, grayscale_cam, save_path):
    """
    Args:
        x_real: (B,3,H,W), 范围[-1,1]
        grayscale_cam: (B,3,H,W) 或 (B,1,H,W)
        save_path: 保存路径
    """

    # [-1,1] -> [0,1]
    imgs = denorm(x_real).detach().cpu()

    overlays = []

    for i in range(imgs.size(0)):
        img = imgs[i].permute(1, 2, 0).numpy()

        if grayscale_cam.size(1) == 3:
            cam = grayscale_cam[i, 0].detach().cpu().numpy()
        else:
            cam = grayscale_cam[i, 0].detach().cpu().numpy()

        overlay = show_cam_on_image(
            img,
            cam,
            use_rgb=True,
            image_weight=0.5,
        )

        overlay = torch.from_numpy(overlay).permute(2, 0, 1).float() / 255.
        overlays.append(overlay)

    overlays = torch.stack(overlays)

    vutils.save_image(overlays, save_path)


def cos_sim(feat1, feat2):
    return torch.mean(F.cosine_similarity(feat1, feat2))