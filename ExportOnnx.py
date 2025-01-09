import torch
import onnx
from efficientnet_pytorch import EfficientNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', help='input image size', default=224)

parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')
opt = parser.parse_args()

path = f'/{opt.volume}/{opt.project}/{opt.subproject}/{opt.task}/{opt.version}'


model_name = f'efficientnet-b{0}'
model = EfficientNet.from_name(model_name, num_classes=2)

model.load_state_dict(torch.load(f"{path}/best.pt"))
model.eval()
device = torch.device("cuda:0")
model = model.to(device)

dummy_input = torch.randn(1, 3, opt.img_size, opt.img_size).to(device)
# batch, channel, size, size

dummy_output = model(dummy_input)

SAVE = f"{path}/best.onnx"

model.set_swish(memory_efficient=False)
torch.onnx.export(model, 
	dummy_input, 
	SAVE, 
	# opset_version=12,
	training=torch.onnx.TrainingMode.EVAL,
	do_constant_folding=True,
	export_params=True,
	verbose=False,)


model = onnx.load(SAVE)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)
