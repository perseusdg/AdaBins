from pickletools import optimize
from models import UnetAdaptiveBins
import model_io
from PIL import Image
import torch
import torch_tensorrt

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256 

model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "./pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)
model = model.cuda()
input = torch.randn([1,3,640,480]).cuda()
model.eval()
output_temp_1,output_temp_2 = model(input)

scripted_torch_model = torch.jit.script(model)
traced_torch_model = torch.jit.trace(model,input,check_trace=True,optimize=True)
torch.jit.save(scripted_torch_model,"scripted_adabins_model.jit")
torch.jit.save(traced_torch_model,"traced_adabins_model.jit")
