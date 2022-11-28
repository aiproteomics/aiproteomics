import os
import tf2onnx

def save_model(
        model, 
        input_format, 
        save_format = 'onnx', 
        output_location = './aiproteomics/modelgen/saved_models/', 
        overwrite = True,
    ):

    # Check save formats    
    valid_save_formats = ['onnx', 'keras']
    invalid_save_formats = []
    if not isinstance(save_format, list):
        save_format = [save_format]
    for format in save_format:
        if format not in valid_save_formats:
            invalid_save_formats.append(format)
        if invalid_save_formats:
            raise ValueError(
                f'Invalid save_format given ({invalid_save_formats}).\n'
                f'Select valid save_format from {valid_save_formats}.'
            )

    if not os.path.exists(output_location):
        os.makedirs(output_location)        

    # Save model
    if input_format == 'keras':
        if 'onnx' in save_format:
            # using default opset and spec settings for now, 
            # might need to be hardcoded if it doesn't work for all cases in the future            
            # for some idea on how to set this, see example on https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
            output_path = output_location + model.name + ".onnx"
            if not overwrite:
                output_path = update_path(output_path)
            tf2onnx.convert.from_keras(model, output_path=output_path)
        if 'keras' in save_format:
            model.save(output_location + model.name)
    else:
        raise NotImplementedError(
            'save_model is currently only implemented for keras models. Other input formats will be added as well.'
        )


def update_path(file):
    basename = os.path.splitext(file)[0]
    ext = os.path.splitext(file)[1]
    n = 1
    
    while os.path.exists(file):
        n += 1
        file = basename + '_' + str(n) + ext
    
    return file
