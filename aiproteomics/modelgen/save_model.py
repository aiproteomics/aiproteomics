import os
import tf2onnx

def save_model(
        model, 
        input_format, 
        save_format = 'onnx', 
        output_location = './aiproteomics/modelgen/saved_models/', 
        overwrite = True,
    ):

    if not isinstance(save_format, list):
        save_format = [save_format]

    if not os.path.exists(output_location):
        os.makedirs(output_location)        

    # Save model
    if input_format == 'keras':
        if 'onnx' in save_format:
            # using default opset and spec settings for now, 
            # might need to be hardcoded if it doesn't work for all cases in the future            
            # for some idea on how to set this, see example on https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
            output_path = output_location + model.name + ".onnx"
            tf2onnx.convert.from_keras(model, output_path=output_path)
        if 'keras' in save_format:
            model.save(output_location + model.name)
    else:
        raise NotImplementedError(
            'save_model is currently only implemented for keras models. Other input formats will be added as well.'
        )

