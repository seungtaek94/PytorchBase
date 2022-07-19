from .hrnet.model import HRNet, HRNet_OCR, HRNet_OCR_B

def get_model(name:str, n_classes:int):
    """ get segmentation model
    Args:
        name (str): _description_
        n_classes (int): _description_
    """

    hrnet = ["hrnet18", "hrnet32", "hrnet48", "hrnet64"]
    hrnet_ocr = ["hrnet18_ocr", "hrnet32_ocr", "hrnet48_ocr", "hrnet64_ocr"]
    hrnet_ocr_b = ["hrnet18_ocr_b", "hrnet32_ocr_b", "hrnet48_ocr_b", "hrnet64_ocr_b"]

    if name in hrnet:
        return HRNet(backbone=name, num_classes=n_classes)
    elif name in hrnet_ocr:
        return HRNet_OCR(backbone=name[:7], num_classes=n_classes)
    elif name in hrnet_ocr_b:
        return HRNet_OCR_B(backbone=name[:7], num_classes=n_classes)
    else:
        raise NotImplementedError

