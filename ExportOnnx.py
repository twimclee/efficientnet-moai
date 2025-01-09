import torch
import onnx
from efficientnet_pytorch import EfficientNet

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', help='image size', type=int, default=224)
parser.add_argument('--nc', help='number of classes', type=int, default=2)
parser.add_argument('--gpu', help='gpu device number', type=int, default=0)

parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')
opt = parser.parse_args()


##########################################################################################
### load .yaml
##########################################################################################

with open(hyp_path, 'r') as file:
    params = yaml.safe_load(file)

    opt.img_size = params.get('img_size')
    opt.nc = params.get('nc')
    opt.gpu = params.get('gpu')

##########################################################################################

path = f'/{opt.volume}/{opt.project}/{opt.subproject}/{opt.task}/{opt.version}/weights'

model_name = f'efficientnet-b{0}'
model = EfficientNet.from_name(model_name, num_classes=opt.nc)

model.load_state_dict(torch.load(f"{path}/best.pt"))
model.eval()
device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")  # set gpu
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
