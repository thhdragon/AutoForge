import types
import torch
import numpy as np
from autoforge.Modules.Optimizer import FilamentOptimizer
import sys

args = types.SimpleNamespace(
    iterations=2,
    warmup_fraction=0.0,
    learning_rate_warmup_fraction=0.0,
    init_tau=0.01,
    final_tau=1.0,
    learning_rate=0.01,
    layer_height=0.04,
    max_layers=4,
    visualize=False,
    tensorboard=False,
    disable_visualization_for_gradio=1,
    output_folder="/tmp",
)

device = torch.device("cpu")
H, W = 16, 16
target = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8).to(torch.float32)
pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
pixel_height_labels = np.zeros((H, W), dtype=np.int32)
global_logits_init = np.zeros((4, 3), dtype=np.float32)
material_colors = torch.rand(3, 3)
material_TDs = torch.ones(3)
background = torch.tensor([0.0, 0.0, 0.0])

print("About to create optimizer with init_tau=0.01, final_tau=1.0")
sys.stdout.flush()

try:
    opt = FilamentOptimizer(
        args=args,
        target=target,
        pixel_height_logits_init=pixel_height_logits_init,
        pixel_height_labels=pixel_height_labels,
        global_logits_init=global_logits_init,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        device=device,
        perception_loss_module=None,
    )
    print("ERROR: Optimizer created successfully, validation did not trigger!")
except ValueError as e:
    print(f"SUCCESS: ValueError raised: {e}")
except Exception as e:
    print(f"UNEXPECTED: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
