import torch

def update_KPI(mask_loss_dict, tao=1):
    """
    Update KPI weights from mask losses.

    Args:
        mask_loss_dict (dict):
            Mapping from model name to mask loss.
        tao (float):
            Focal exponent.

    Returns:
        dict:
            Mapping from model name to KPI weight.
    """
    with torch.no_grad():

        kpi_dict = {}

        for model_name, loss in mask_loss_dict.items():

            loss = torch.clamp(
                torch.as_tensor(loss),
                min=1e-13,
                max=1.0,
            )

            kpi = -(1 - loss).pow(tao) * torch.log(loss)

            kpi_dict[model_name] = kpi

        return kpi_dict