import torch
import onnx
from efficientnet_pytorch import EfficientNet

import argparse


from pathfilemgr import MPathFileManager
from hyp_data import MHyp, MData

parser = argparse.ArgumentParser()
parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')
opt = parser.parse_args()


mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)
mhyp = MHyp()
mdata = MData()
mpfm.load_test_hyp(mhyp)
mpfm.load_test_data(mdata)


model_name = f'efficientnet-b{mhyp.model}'
model = EfficientNet.from_name(model_name, num_classes=mhyp.num_class)

model.load_state_dict(torch.load(f"{mpfm.weight_path}/best.pt"))
model.eval()
device = torch.device(f"cuda:{mhyp.gpu}" if torch.cuda.is_available() else "cpu")  # set gpu
model = model.to(device)

dummy_input = torch.randn(1, 3, mhyp.img_size, mhyp.img_size).to(device)
# batch, channel, size, size

dummy_output = model(dummy_input)

SAVE = f"{mpfm.weight_path}/best.onnx"

model.set_swish(memory_efficient=False)
torch.onnx.export(model, 
	dummy_input, 
	SAVE, 
	opset_version=12,
	training=torch.onnx.TrainingMode.EVAL,
	do_constant_folding=True,
	export_params=True,
	verbose=False,)

model = onnx.load(SAVE)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)
