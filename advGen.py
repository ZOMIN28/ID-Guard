import os
import torch
from torch.autograd import Variable
from net.advGen import ResnetGenerator, weights_init

class advGenerator:
    def __init__(self, device, if_pretrained, model_save_path, epsilon=0.05):
        
        self.device = device
        
        self.epsilon = epsilon

        self.generator = ResnetGenerator(norm_type='batch',input_nc=6).to(self.device)
        self.generator.apply(weights_init)

        # not use pretrained
        if not if_pretrained:
            self.pri_pert = torch.load("checkpoint/ID-guard/pri_pert.pt")
            self.pri_pert = torch.clamp(self.pri_pert, -self.epsilon, self.epsilon)
            self.pri_pert = Variable(self.pri_pert.to(self.device), requires_grad=True)
        
        # use pretrained
        else:
            self.pri_pert = torch.load(model_save_path["pert_path"])
            self.pri_pert = torch.clamp(self.pri_pert, -self.epsilon, self.epsilon)
        
            self.generator.load_state_dict(torch.load(model_save_path["model_path"], map_location="cpu"))
            self.generator.eval()


    def generate(self, x_input):
        
        # prior perturbation
        pri_pert = torch.clamp(self.pri_pert, -self.epsilon, self.epsilon)

        # cat the input image and prior perturbation
        x_cat = torch.cat((x_input, pri_pert.repeat(x_input.size(0),1,1,1)), 1)
        
        # generate perturbation
        perturbation = self.generator(x_cat)
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        
        # get adversarial image 
        x_adv = torch.clamp(x_input + perturbation, -1.0, 1.0)
        x_adv_pri = torch.clamp(x_input + pri_pert, -1.0, 1.0)
        
        return x_adv, x_adv_pri