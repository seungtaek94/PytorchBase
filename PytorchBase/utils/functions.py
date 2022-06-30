import torch
import numpy as np
import random
from typing import List, Tuple


def seed_everything(seed:int): # seed 고정
    """ seed everything for pytorch

    Args:
        seed (int): 
            seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(tensor:torch.Tensor) -> np.ndarray:
    """convert torch.Tensor to numpy.ndarray.

    Args:
        tensor (torch.Tensor): 
            Input tensor to convert.

    Returns:
        np.ndarray: 
            Output result.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


ort_type2numpy_type = {
    'double': np.float64,
    'float': np.float32,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64
}


def allclose_iter(src1:np.ndarray, src2:np.ndarray, 
    rtol:float=1e-3, atol:float=1e-5, gamma:int=10, iter:int=2) -> Tuple[bool, int]:
    """ Compare the two numpy arrays(`src1`, `src2`) `iter` times, increasing `atol`.
        Comapaere rule:
            `absolute(src1 - src2) <= (atol + rtol * absolute(src2))`      

    Args:
        src1 (np.ndarray): 
            Input arrays to compare.
        src2 (np.ndarray): 
            Input arrays to compare.
        rtol (float, optional): 
            The relative tolerance parameter. 
        atol (float, optional): 
            The relative tolerance parameter. 
        gamma (int, optional):
            If failed to pass comparing then increase `atol` at next step flowing rule:
                `atol = atol * gamma`. Defaults to 10.
        iter (int, optional): 
            Iteration for comparing. Defaults to 2.

    Returns:
        Tuple[bool, int]: 
            Compare result [result(True or False), passed or failed step]
    """
    for i in range(iter):
        if np.allclose(src1, src2, rtol=rtol, atol=atol):
            return True, i
        else:
            atol = atol * gamma
    
    return False, iter


def check_onnx_model(
    torch_model, 
    ort_session, 
    input_tensors:List[torch.Tensor or np.ndarray]=None, 
    rtol=1e-3, 
    atol=1e-5, 
    gamma=10, 
    iter=2) -> bool:
    """Check constancy of both torch_model and ort_session predict results.

    Args:
        torch_model (_type_): 
            Torch model.
        ort_session (_type_): 
            Onnxruntime session.
        input_tensors (List[torch.Tensor or np.ndarray], optional): 
            Input dummy tensor. If `input_tensors==None` it will be create
            dummy tensor that has same shape with the onnxruntime sesstion input shape automatically.
            If onnxruntime session has dynamic input shape it will be failed to compare
            then you can give dummy input tensor which has fixed shape to this parameter.            
            Defaults to None.
        rtol (float, optional): 
            The relative tolerance parameter. 
        atol (float, optional): 
            The relative tolerance parameter. 
        gamma (int, optional): 
            If failed to pass comparing then increase `atol` at next step flowing rule:
                `atol = atol * gamma`. Defaults to 10.
        iter (int, optional): 
            Iteration for comparing. Defaults to 2.

    Raises:
        Exception: Doesn't match input_tensors length with number of model inputs.
        Exception: Unsupported tensortype.
        Exception: Number of onnx model predict results is not same with ort_session.get_outputs().
        Exception: The output values of the both models are different.

    Returns:
        bool: if pass return `True` otherwise return `False`.
    """

    input_torch_tensors = []
    input_ort_tensors = {}

    if input_tensors != None:
        if len(input_tensors) != len(ort_session.get_inputs()):
            raise Exception("Doesn't match input_tensors length with number of model inputs !!")

        for i, tensor in enumerate(input_tensors):
            if type(tensor) == torch.Tensor:
                input_torch_tensors.append(tensor)
                input_ort_tensors[ort_session.get_inputs()[i].name] = \
                    to_numpy(tensor)
            elif type(tensor) == np.ndarray:
                input_torch_tensors.append(torch.from_numpy(tensor))
                input_ort_tensors[ort_session.get_inputs()[i].name] = \
                    tensor
            else:
                raise Exception("Unsupported tensor type !!")
    else:
        for input_info in ort_session.get_inputs():        
            tensor_type = str(input_info.type).split('(')[1].split(')')[0]

            if tensor_type not in ort_type2numpy_type.keys():
                raise Exception(f"Unsupport tensortype !! - {tensor_type}")

            input_ort_tensors[input_info.name] = np.random.rand(*input_info.shape)
            
            if str(input_info.type) in "int":
                input_ort_tensors[input_info.name] = input_ort_tensors[input_info.name] * 255
            
            input_ort_tensors[input_info.name] = \
                 input_ort_tensors[input_info.name].astype(ort_type2numpy_type[tensor_type])

            input_torch_tensors.append(torch.from_numpy(input_ort_tensors[input_info.name]))

    torch_model.eval()
    torch_outputs = torch_model(*input_torch_tensors)
    ort_outputs = ort_session.run(None, input_ort_tensors)

    if len(ort_outputs) != len(ort_session.get_outputs()):
        raise Exception(
            "Number of onnx model predict results \
            is not same with ort_session.get_outputs().")
    
    if len(ort_outputs) == 1:
        result, step = allclose_iter(
            to_numpy(torch_outputs), ort_outputs[0], rtol=rtol, atol=atol, gamma=gamma, iter=iter)
        
        if result == True:
            if step != 0:
                step_str =  f"{step+1}nd" if step == 1 else f"{step+1}th"                    
                print(f"[Warnning] Passed output tensor check. But pass at {step_str} step. \
                    (atol={atol * (gamma ** step)})")
            return True
        else:
            raise Exception("The output values of the both models are different!!")
    
    else:
        results = []
        for i in range(len(ort_outputs)):
            results.append(allclose_iter(
                to_numpy(torch_outputs[i]), ort_outputs[i], rtol=rtol, atol=atol, gamma=gamma, iter=iter))
        
        for i, (result, step) in enumerate(results):
            result_str = "Passed" if result else "Failed"

            if step == 0:
                step_str = "1st"
            elif step == 1:
                step_str = "2nd"
            else:
                step_str = f"{step+1}th"

            print(f"Output {i}: {result_str} at {step_str} step !!")
            
            if result == False:
                raise Exception("The outputs values of the both models are different!!")
        return True