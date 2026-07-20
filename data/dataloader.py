from data.dataset import get_loader

"""
------------------------------
   get dataloader
------------------------------
"""
def getDataloader(config, selected_attrs=None, shuffle=14):
    
    if selected_attrs == None:
        selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    
    return get_loader(config, selected_attrs=selected_attrs, shuffle=shuffle)