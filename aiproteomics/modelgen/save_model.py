import os
import tf2onnx

def save_model(
        model,
        input_format,
        save_format, 
        output_destination,
        overwrite,
        ):
    # Save model
    output_location = './aiproteomics/modelgen/saved_models/'
    if save_format == None:
        # do not save
        pass
    else:
        if save_format == 'onnx' or save_format == 'both':
            # save as onnx
            output_path = output_location + model.name + ".onnx"
            # using default opset and spec settings for now, might need to be hardcoded if it doesn't work for all cases in the future            
            # for some idea on how to set this, see example on https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
            tf2onnx.convert.from_keras(model, output_path=output_path)
        if save_format == 'keras' or save_format == 'both':
            # save as keras
            model.save(output_location + model.name)