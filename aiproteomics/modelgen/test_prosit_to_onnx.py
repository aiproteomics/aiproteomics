from onnx import load
from .prosit1_gen import build_prosit1_model
import os

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

def test_prosit():
    build_prosit1_model()

    # leaving below code in, because we'll use parts of this to get the model parameters
    # see https://onnx.ai/onnx/intro/python.html for more info
    
    model_path = './aiproteomics/modelgen/saved_models/model.onnx'

    with open(model_path, "rb") as f:
        onnx_model = load(f)
        # # display
        # # in a more nicely format
        # print('** inputs **')
        # for obj in onnx_model.graph.input:
        #     print("name=%r dtype=%r shape=%r" % (
        #         obj.name, obj.type.tensor_type.elem_type,
        #         shape2tuple(obj.type.tensor_type.shape)))
        
        # for node in onnx_model.graph.node:
        #     print("name=%r type=%r input=%r output=%r" % (
        #     node.name, node.op_type, node.input, node.output))
    
        assert len(onnx_model.graph.input) == 3
    