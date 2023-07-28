# use onnxruntime to directly inference onnx model
import onnx
import onnxruntime as ort
import cv2
import time
import numpy as np

model_path = "model_zoo/rtmpose-ort/rtmdet-nano/end2end.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)    # check the model

input = cv2.imread('assets/onnx_test.jpg')
input = cv2.resize(input, (640, 640))
input = input.astype('float32')

# reshape input from (H, W, 3) to (1, 3, H, W)
input = input.transpose((2, 0, 1))
input = input[np.newaxis, :]

# build session
providers = ['CPUExecutionProvider']
sess = ort.InferenceSession(model_path, providers=providers)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
names = (output_name, input_name)

# test spped
start = time.time()
for i in range(50):
    outputs = sess.run([output_name], {input_name: input})
end = time.time()
# diable science notation
np.set_printoptions(suppress=True)
print(outputs[0])
print(outputs[0].shape)

print(f'Inference takes {(end - start) * 20:.2f} ms')


# test inference speed of PoseInferencerV3
from tools.inferencer import PoseInferencerV3
import mmengine
import cv2

from easydict import EasyDict

args = EasyDict()
args.config = 'configs/mark3.py'

img = 'assets/onnx_test.jpg'
img = cv2.imread(img)
cfg = mmengine.Config.fromfile(args.config)

inferencer = PoseInferencerV3(cfg.det_cfg, cfg.pose_cfg)

start = time.time()
for i in range(50):
    outputs = inferencer.detector(img)
end = time.time()
print(f'Inference takes {(end - start) * 20:.2f} ms')