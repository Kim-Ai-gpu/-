import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

if __name__ == '__main__':

    elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo"]

    input_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

 
    model = models.resnet50()
    model.fc = nn.Linear(2048, 99) 
    model.load_state_dict(torch.load('weights.pth', map_location='cpu'))
    model.eval()


    img = Image.open('M57.png')


    img_tensor = input_transform(img).unsqueeze(0)


    with torch.no_grad():
        result = model(img_tensor)
        result = torch.sigmoid(result)

    sorted_values, sorted_indices = torch.sort(result, descending=True)

    for idx, value in zip(sorted_indices[0], sorted_values[0]):
        print(f"Element: {elements[idx.item()]}, Probability: {value.item()}")
