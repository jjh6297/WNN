import onnx
import torch.onnx
import tf2onnx
import tensorflow as tf
import model as wnn_pt
import WNN as wnn_tf
import os

script_dir = os.path.dirname(os.path.abspath('onnx_export.py'))

def import_onnx_model_WNN_tf():
    model = wnn_tf.WNN(5)
    model.load_weights(filepath='WNN_PT/NWNN_Conv_13.h5')
    spec = (tf.TensorSpec((1,5), tf.float32, name="input_1"), tf.TensorSpec((1,4), tf.float32, name="input_2")) # input signature
    file_path = os.path.join(script_dir, "model_tf.onnx")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=file_path)


def import_onnx_model_WNN_pt():
    file_path = os.path.join(script_dir, "model_pt.onnx")
    model = wnn_pt.WNN()
    x = torch.randn(1, 5)
    x2 = torch.index_select(x, 1, torch.tensor([0,1,2,3])) - torch.index_select(x, 1, torch.tensor([1,2,3,4]))
    # [x, x2] represents dummy inputs for the model to be exported
    torch.onnx.export(model, [x,x2], file_path)