import os
import common

default_model_dir = '../all_models'
default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
model = os.path.join(default_model_dir, default_model)
interpreter = common.make_interpreter(model)
# interpreter.allocate_tensors()

width, height, channels = common.input_image_size(interpreter)
print(f"""
width={width}
height={height}
channels={channels}
""")