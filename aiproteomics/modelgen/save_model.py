import os
import tf2onnx


VALID_OUTPUT_FORMATS = ['onnx', 'keras']


def save_model( # pylint: disable=too-many-arguments
        model: str, 
        name: str,
        framework: str = 'keras',
        output_formats: str = 'onnx',
        output_dir: str = './aiproteomics/modelgen/saved_models/',
        overwrite: bool = True,
    ):

    """Function to save the model.

    Args:
        model (str): the model to be saved
        
        name (str): the name to save the model as

        framework (str): the framework used to generated the model.
            Currently only keras is implemented.

        output_formats (list or str): format or list of formats to save the model as.
            Currently only 'onnx' and 'keras' are implemented.
            Defaults to 'onnx'.

        output_dir (str): directory to save the model

        overwrite (bool): 
            if True: any existing file model with the same name will be overwritten.
            if False: will append a number behind the file name to create a new file
    """

    # Check save formats    
    invalid_output_formatss = []
    if not isinstance(output_formats, list):
        output_formats = [output_formats]
    for output_format in output_formats:
        if output_format not in VALID_OUTPUT_FORMATS:
            invalid_output_formatss.append(output_format)
        if invalid_output_formatss:
            raise ValueError(
                f'Invalid output_formats given ({invalid_output_formatss}).\n'
                f'Select valid output_formats from {VALID_OUTPUT_FORMATS}.'
            )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)        

    # Save model
    if framework == 'keras':
        if 'onnx' in output_formats:
            # using default opset and spec settings for now, 
            # might need to be hardcoded if it doesn't work for all cases in the future            
            # for some idea on how to set this, see example on https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
            output_path = output_dir + name + ".onnx"
            if not overwrite:
                output_path = update_path(output_path)
            tf2onnx.convert.from_keras(model, output_path=output_path)
        if 'keras' in output_formats:
            model.save(output_dir + name)
    else:
        raise NotImplementedError(
            'save_model is currently only implemented for keras models. Other input formats will be added as well.'
        )


def update_path(file):
    """
    Adds a number behind the filename if it already exists to avoid overwriting the same file.
    """
    
    basename = os.path.splitext(file)[0]
    ext = os.path.splitext(file)[1]
    n = 1
    
    while os.path.exists(file):
        n += 1
        file = basename + '_' + str(n) + ext
    
    return file
