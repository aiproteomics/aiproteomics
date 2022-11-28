import os
import tf2onnx

def save_model(
        model, 
        name,
        framework = 'keras',
        output_format = 'onnx',
        output_dir = './aiproteomics/modelgen/saved_models/',
        overwrite = True,
    ):

    """Function to save the model.

    Args:
        model (str): the model to be saved
        
        name (str): the name to save the model as

        framework (str): the framework used to generated the model.
            Currently only keras is implemented.

        output_format (list or str): format or list of formats to save the model as.
            Currently only 'onnx' and 'keras' are implemented.
            Defaults to 'onnx'.

        output_dir (str): directory to save the model

        overwrite (bool): 
            if True: any existing file model with the same name will be overwritten.
            if False: will append a number behind the file name to create a new file
    """

    # Check save formats    
    valid_output_formats = ['onnx', 'keras']
    invalid_output_formats = []
    if not isinstance(output_format, list):
        output_format = [output_format]
    for format in output_format:
        if format not in valid_output_formats:
            invalid_output_formats.append(format)
        if invalid_output_formats:
            raise ValueError(
                f'Invalid output_format given ({invalid_output_formats}).\n'
                f'Select valid output_format from {valid_output_formats}.'
            )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)        

    # Save model
    if framework == 'keras':
        if 'onnx' in output_format:
            # using default opset and spec settings for now, 
            # might need to be hardcoded if it doesn't work for all cases in the future            
            # for some idea on how to set this, see example on https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
            output_path = output_dir + name + ".onnx"
            if not overwrite:
                output_path = update_path(output_path)
            tf2onnx.convert.from_keras(model, output_path=output_path)
        if 'keras' in output_format:
            model.save(output_dir + name)
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
