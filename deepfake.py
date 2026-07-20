import torch
from PIL import Image
import json
import os
from pathlib import Path
from torchvision import transforms

# target model
import model.stargan as stargan
from model.HiSD.trainer import HiSD_Trainer
import model.fpgan as fpgan
import model.attentiongan as attentiongan
import model.relgan as relgan

from utils.utils import create_labels, get_config
from utils.attack import LinfPGDAttack


class DeepfakeHandler:
    def __init__(self, device, deepfake_model_path, model_type="stargan"):
        """
        model_type: str, one of ["stargan", "aggan", "fpgan", "relgan", "HiSD"]
        """
        with open(deepfake_model_path, "r") as f:
            self.model_cfg = json.load(f)
            
        self.model_type = model_type
        self.device = device
        self.model = self._load_model(model_type)

        print(f"==== Model loaded successfully: {self.model_type} =====")


    """
    ------------------------------------
            Load model weights
    ------------------------------------
    """
    def _load_checkpoint(self, model, checkpoint_path):
        """
        Load model weights.

        Args:
            model: PyTorch model.
            checkpoint_path (str): Path to checkpoint.

        Returns:
            Loaded model.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

        model = model.to(self.device)

        return model
    
    """
    ------------------------------------
        Build model architecture only
    ------------------------------------
    """
    def _build_model(self, model_type):

        builders = {
            "stargan": lambda: stargan.Generator(conv_dim=64, c_dim=5, repeat_num=6),
            "aggan": lambda: attentiongan.Generator(),
            "fpgan": lambda: fpgan.Generator(conv_dim=64, c_dim=5, repeat_num=6),
            "relgan": lambda: relgan.G(), 
        }

        if model_type not in builders:
            raise ValueError(f"Unsupported model type: {model_type}")

        return builders[model_type]()


    """
    ------------------------------------
                Model Loading
    ------------------------------------
    """
    def _load_model(self, model_type):
        """
        Load deepfake model.

        Args:
            model_type (str): Name of deepfake model.

        Returns:
            Loaded model.
        """

        # prcossing HiSD 
        if model_type == "HiSD":

            config = get_config(self.model_cfg["HiSD"]["config"])
            self.hisd_config = config

            trainer = HiSD_Trainer(config)

            checkpoint = torch.load(self.model_cfg["HiSD"]["checkpoint"], map_location="cpu")

            trainer.models.gen.load_state_dict(checkpoint["gen_test"])

            trainer.models.gen = trainer.models.gen.to(self.device)

            return trainer.models.gen

        # processing stargan, aggan, fpgan, relgan
        model = self._build_model(model_type)

        checkpoint_path = self.model_cfg[model_type]["checkpoint"]

        model = self._load_checkpoint(model, checkpoint_path)

        # !! only relgan need .eval()
        if model_type == "relgan":
            model.eval()

        return model

    
    """
    ------------------------------------
       Pre-processing the references
    ------------------------------------
    """
    def process_ref(self, c_org=None, selected_attrs=None):
        """
        Prepare reference inputs for different deepfake models.

        Args:
            c_org (Tensor, optional):
                Original attribute labels.
            selected_attrs (list[str], optional):
                Attribute names used by StarGAN-based models.

        Returns:
            Reference inputs required by the selected deepfake model.
        """

        if selected_attrs is None:
            selected_attrs = [
                "Black_Hair",
                "Blond_Hair",
                "Brown_Hair",
                "Male",
                "Young",
            ]

        # ---------------------------------------------------------
        # StarGAN / AGGAN / FPGAN
        # ---------------------------------------------------------
        if self.model_type in {"stargan", "aggan", "fpgan"}:
            return create_labels(c_org, len(selected_attrs), selected_attrs)

        # ---------------------------------------------------------
        # RelGAN
        # ---------------------------------------------------------
        if self.model_type == "relgan":

            attributes = [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0],
            ]

            return [torch.tensor(attr, device=self.device).unsqueeze(0) for attr in attributes]

        # ---------------------------------------------------------
        # HiSD
        # ---------------------------------------------------------
        if self.model_type == "HiSD":

            image_size = self.hisd_config["new_size"]

            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            reference_dir = Path("model/HiSD/examples")

            image_paths = sorted(
                [
                    p for p in reference_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]
            )

            references = []

            for image_path in image_paths:

                filename = image_path.stem.lower()

                if "glasses" in filename:
                    attr_id = 1
                elif "haircolor" in filename:
                    attr_id = 2
                else:
                    attr_id = 0

                ref_img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)

                references.append([attr_id, ref_img])

            return references

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


    """
    ------------------------------------
              Manipulation
    ------------------------------------
    """
    def manipulate(self, img, c_ref):
        if self.model_type == "stargan":
            return self.model(img, c_ref)

        elif self.model_type == "aggan":
            gen_img, _, _ = self.model(img, c_ref)
            gen_img
            return gen_img
        
        elif self.model_type == "fpgan":
            return torch.tanh(img + self.model(img, c_ref))
        
        elif self.model_type == "relgan":
            c_ref = c_ref.repeat(img.size(0), 1)
            return self.model(img, c_ref)


        elif self.model_type == "HiSD":
            type_num, r = c_ref
            return self.model(img, r, type_num)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


    """
    ------------------------------------
     get mid feature of deepfake model
    ------------------------------------
    """
    def get_feature(self, img, c_ref):
        if self.model_type in ["stargan", "aggan", "fpgan"]:
            return self.model.mid_features(img, c_ref)
        
        elif self.model_type == "relgan":
            c_ref = c_ref.repeat(img.size(0), 1)
            return self.model.mid_features(img, c_ref)

        elif self.model_type == "HiSD": 
            return self.model.encode(img)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")



    """
    --------------------------------------------
        Proactive Defense against Deepfake
    --------------------------------------------
    """
    def proactive_defend(self, img, c_ref):
        """
        PGD/I-FGSM attack
        """
        attack = LinfPGDAttack(handler=self, device=self.device, epsilon=0.05)
        with torch.no_grad():
            output = self.manipulate(img, c_ref)
        x_adv, _ = attack.perturb(img, output, c_ref)
        
        return x_adv