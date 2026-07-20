# common
import os
import json
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchvision.utils import save_image
import pytorch_grad_cam 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# dynamic weighting strategy 
from strategy.MGDA import update_MGDA
from strategy.KPI import update_KPI

# models
from advGen import advGenerator
from net.vggface import vggface
from net.arcface import ResNet as arcface
from deepfake import DeepfakeHandler

# dataset
from data.dataloader import getDataloader

# utils
from utils.utils import denorm, resize_image, cos_sim, save_heatmap_overlay, Image2tensor



class Solver:
    def __init__(self, config):
        
        # init
        for k, v in vars(config).items():
            setattr(self, k, v)

        self.loss_fn = F.mse_loss
        self.iter = 0
        self.scaler = torch.cuda.amp.GradScaler()
        self._create_dir()

        # get dataloader
        self.data_loader = getDataloader(config)

        # get deepfake model depend on tar_model list
        self.model_dict = {model_type : DeepfakeHandler(self.device, self.deepfake_model_path, model_type) 
                              for model_type in self.target_model}
        for model in self.model_dict.values():
            for p in model.model.parameters():
                p.requires_grad = False
        
        # Preprocess the reference image for HiSD to reduce redundant calculations
        if "HiSD" in self.target_model:
            self.hisd_ref = self.model_dict["HiSD"].process_ref()

        # load id model
        if self.mode == "train":
            with open(self.id_model_path, "r") as f:
                id_model_config = json.load(f)
            self._load_idmodel(id_model_config)

        # init adv generator
        if_pretraind = False if self.mode == "train" else True
        self.advG = advGenerator(self.device, if_pretraind, self.model_save_path, self.epsilon)

        # init optimizer
        if self.mode == "train":
            self.optimizer_G = torch.optim.Adam(list(self.advG.generator.parameters()) + 
                                            list([self.advG.pri_pert]), self.lr, [self.beta1, self.beta2])


        # init prior weights
        with open(self.prior_weight_path, "r") as f:
            dws_config = json.load(f)
        try:
            self.pri_weight = dws_config[config.dws]
        except KeyError:
            raise ValueError(
                f"Unsupported DWS strategy: {config.dws}"
            )
        
    
    def _create_dir(self):
        # create basic directories
        for directory in [self.model_save_dir, self.sample_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)

        # create model-specific directory
        model_dir = os.path.join(
            self.model_save_dir,
            f"{self.dws}_{self.id_extractor}"
        )
        os.makedirs(model_dir, exist_ok=True)

        self.model_save_path = {
            "model_path": os.path.join(model_dir, "AdvG.pth"),
            "pert_path": os.path.join(model_dir, "optimized_pri_pert.pt"),
        }


    def _load_idmodel(self, id_model_config):
        # load vggface
        self.vggface = vggface().to(self.device)
        self.vggface.load_state_dict(torch.load(id_model_config["vggface"]))
        self.vggface.eval()

        if self.id_extractor == "Arc":
            # load arcnet
            self.arcnet = arcface().to(self.device)
            self.arcnet.load_state_dict(torch.load(id_model_config["arcnet"]))
            for p in self.arcnet.parameters():
                p.requires_grad = False
            self.arcnet.eval()

            self.id_resize = 112
            self.id_model = self.arcnet.features
        
        elif self.id_extractor  == "Vgg":
            self.id_resize = 224
            self.id_model = self.vggface.features
        
        else:
            raise ValueError(f"Unsupported id_extractor: {self.id_extractor}")
        
    
    def heatmap(self, x_real):
        for p in self.vggface.parameters():
                p.requires_grad = True

        traget_layers = [self.vggface.conv5]
        cam = pytorch_grad_cam.GradCAMPlusPlus(model=self.vggface, target_layers=traget_layers, use_cuda=True)
        # Resize to match input requirement
        resized_input = resize_image(denorm(x_real), 224)
    
        with torch.no_grad():
            outputs = self.vggface(resized_input.to(self.device))
            pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()

        targets = [ClassifierOutputTarget(c) for c in pred_classes]
        grayscale_cam = cam(resized_input, targets=targets)
        grayscale_cam = torch.tensor(grayscale_cam).unsqueeze(1).repeat(1, 3, 1, 1)
        grayscale_cam = resize_image(grayscale_cam, x_real.size(2))
        
        # save_heatmap_overlay(x_real, grayscale_cam, os.path.join(self.sample_dir, "heatmap.jpg"))
        return grayscale_cam.to(self.device)


    def _feat_loss(self, feat_ori, feat_adv, mask):
        mask = resize_image(mask[:, :1].repeat(1, feat_ori.size(1), 1, 1), feat_ori.size(2))
        return F.mse_loss(feat_ori * mask, feat_adv * mask)


    def _mask_loss(self, output_ori, output_adv, mask, hmp):
        if self.mask_type == "both":
            # add
            if self.mask_comb_type == "add":
                return F.mse_loss(output_ori * mask, output_adv * mask) + \
                    F.mse_loss(output_ori * hmp, output_adv * hmp)
            
            # multi
            if self.mask_comb_type == "multi":
                return F.mse_loss(output_ori * mask * hmp, output_adv * mask * hmp)
            
            raise ValueError(f"Unsupported mask_comb_type: {self.mask_comb_type}")

        if self.mask_type == "bin":
            return F.mse_loss(output_ori * mask, output_adv * mask)

        if self.mask_type == "hmp":
            return F.mse_loss(output_ori * hmp, output_adv * hmp)

        raise ValueError(f"Unsupported mask_type: {self.mask_type}")


    def adv_loss(self, output_ori_resize, output_adv_resize, feat_ori, feat_adv, 
                 output_ori, output_adv, mask, hmp):

        mask_loss = self._mask_loss(output_ori, output_adv, mask, hmp)

        id_loss = -cos_sim(
            self.id_model(output_ori_resize),
            self.id_model(output_adv_resize),
        )

        ft_loss = self._feat_loss(feat_ori, feat_adv, mask)

        loss = (
            self.lambda_mask * mask_loss
            + self.lambda_id * id_loss
            + self.lambda_ft * ft_loss
        )

        return loss, mask_loss.item()

    def count_loss(self, x_real, mask, hmp, output_ori, output_adv, output_adv_pri,
               feat_ori, feat_adv, feat_adv_pri):

        output_ori_resize = resize_image(output_ori, self.id_resize)
        output_adv_resize = resize_image(output_adv, self.id_resize)
        output_adv_pri_resize = resize_image(output_adv_pri, self.id_resize)

        adv_loss, mask_loss = self.adv_loss(
            output_ori_resize,
            output_adv_resize,
            feat_ori,
            feat_adv,
            output_ori,
            output_adv,
            mask,
            hmp,
        )

        adv_loss_pri, mask_loss_pri = self.adv_loss(
            output_ori_resize,
            output_adv_pri_resize,
            feat_ori,
            feat_adv_pri,
            output_ori,
            output_adv_pri,
            mask,
            hmp,
        )

        return adv_loss + adv_loss_pri, mask_loss + mask_loss_pri
    

    def train_batch(self, x_real, mask, c_org):
        self.advG.generator.train()
        
        # init
        total_loss = 0.
        loss_dict = {model: 0.0 for model in self.target_model}
        mask_loss_dict = {model: 0.0 for model in self.target_model}
        grads = {model: 0.0 for model in self.target_model}

        # get heatmap of images in vggface
        hmp = self.heatmap(x_real).detach()
        for p in self.vggface.parameters():
                p.requires_grad = False
        
        with torch.cuda.amp.autocast():
            # get adversarial images
            x_adv, x_adv_pri = self.advG.generate(x_real)

            # to every target deepfake model
            for _, model in enumerate(self.target_model):
                # get reference or label
                references = self.hisd_ref if model == "HiSD" else self.model_dict[model].process_ref(c_org)
                ref = references[self.iter % len(references)]
                
                # get orignal output
                with torch.no_grad():
                    output_ori = self.model_dict[model].manipulate(x_real, ref)
                    feat_ori = self.model_dict[model].get_feature(x_real, ref)

                # get adversarial output
                output_adv = self.model_dict[model].manipulate(x_adv, ref)
                output_adv_pri = self.model_dict[model].manipulate(x_adv_pri, ref)

                # get adversarial mid feature output    
                feat_adv = self.model_dict[model].get_feature(x_adv, ref)
                feat_adv_pri = self.model_dict[model].get_feature(x_adv_pri, ref)
                
                # get loss
                model_loss, mask_loss_dict[model] = self.count_loss(x_real, mask, hmp, output_ori, output_adv, output_adv_pri, feat_ori, feat_adv, feat_adv_pri)
                loss_dict[model] = model_loss
                
                # if MGDA, get grads of every deepfake model attack 
                if self.dws == "MGDA":
                    grad = torch.autograd.grad(model_loss, self.advG.generator.parameters(), retain_graph=True, create_graph=False)
                    grads[model] = [g.detach().clone() if g is not None else torch.zeros_like(p) 
                                        for g, p in zip(grad, self.advG.generator.parameters())] 

                with torch.no_grad():
                    # save result
                    if self.iter % self.save_iter == 0:
                        save_list = []
                        for tensor_group in [x_real, output_ori, output_adv, output_adv_pri]:
                            save_list.extend([denorm(i).cpu() for i in tensor_group])
                        
                        save_path = os.path.join(self.sample_dir, f"sample_{model}.jpg")
                        save_image(torch.stack(save_list), save_path, nrow=self.batch_size)   
                        save_list.clear()
        
            with torch.no_grad():
                # if MGDA, update dynamic weights
                if self.dws == "MGDA":
                    dyn_weight = update_MGDA(grads)
            
                # if KPI, update dynamic weights
                elif self.dws == "KPI":
                    dyn_weight = update_KPI(mask_loss_dict)
        
            # get the total loss of every deepfake model attack
            for _, model in enumerate(self.target_model):
                total_loss += loss_dict[model] * self.pri_weight[model] * dyn_weight[model]

        # update
        G_loss = - total_loss
        self.optimizer_G.zero_grad()
        self.scaler.scale(G_loss).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        return loss_dict, dyn_weight



    def train(self): 
        for epoch in range(1, self.epochs + 1):
            loss = {model: 0.0 for model in self.target_model}

            with tqdm(total=len(self.data_loader), desc=f"Epoch {epoch}/{self.epochs}", unit="batch") as pbar:

                for idx, (img, mask, c_org) in enumerate(self.data_loader):

                    img = img.to(self.device)
                    mask = mask.to(self.device)
                    c_org = c_org.to(self.device)

                    loss_dict, dyn_weight = self.train_batch(img, mask, c_org)

                    with torch.no_grad():
                        for model, value in loss_dict.items():
                            loss[model] += value.item()

                    postfix = {}
                    for model in self.target_model:
                        postfix[f"{model} loss:"] = f"{loss[model] / (idx + 1):.4f}"
                        # postfix[f"{model} weight:"] = f"{dyn_weight[model]:.3f}"

                    pbar.set_postfix(postfix)
                    pbar.update()

                    self.iter += 1

            # save generator and optimized prior perturbation
            torch.save(self.advG.generator.state_dict(), self.model_save_path["model_path"])
            torch.save(self.advG.pri_pert, self.model_save_path["pert_path"])



    def test(self):
        with torch.no_grad():  
            self.advG.generator.eval()
            for idx, (x_real, mask, c_org) in enumerate(tqdm(self.data_loader)):
                x_real, mask, c_org = x_real.to(self.device), mask.to(self.device), c_org.to(self.device)
                x_adv, x_adv_pri = self.advG.generate(x_real)

                for _, model in enumerate(self.target_model):
                    # get reference or label
                    references = self.hisd_ref if model == "HiSD" else self.model_dict[model].process_ref(c_org)
                    
                    rows = [[] for _ in range(4)]
                    for ref in references:
                        # get orignal output
                        output_ori = self.model_dict[model].manipulate(x_real, ref)

                        # get adversarial output
                        output_adv = self.model_dict[model].manipulate(x_adv, ref)
                        output_adv_pri = self.model_dict[model].manipulate(x_adv_pri, ref)
                        
                        rows[0].extend([denorm(img).cpu() for img in x_real])
                        rows[1].extend([denorm(img).cpu() for img in output_ori])
                        rows[2].extend([denorm(img).cpu() for img in output_adv])
                        rows[3].extend([denorm(img).cpu() for img in output_adv_pri])

                    # save_result
                    save_list = []
                    for row in rows:
                        save_list.extend(row)
                    
                    save_path = os.path.join(self.results_dir, f"result_{idx}_{model}.jpg")
                    save_image(torch.stack(save_list), save_path, nrow=self.batch_size)   
                
                
    
    def test_one(self):
        with torch.no_grad():
            x_input = Image2tensor(self.test_image_path)  
            self.advG.generator.eval()
            x_real = x_input.to(self.device)
            x_adv, x_adv_pri = self.advG.generate(x_real)

            return x_adv, x_adv_pri
            