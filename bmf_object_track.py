import bmf
import os
import torch

GROUNDING_DINO_MODEL_PATH = "/home/pamin/bmf_seg/gsam2/gdino_checkpoints/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = "/home/pamin/bmf_seg/gsam2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
SAM2_MODEL_PATH = "/home/pamin/bmf_seg/gsam2/checkpoints/sam2.1_hiera_small.pt"
SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_s.yaml"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
input_video_path = "/home/pamin/bmf_seg/car_tracking_input.mp4"
output_video_path = "./car_tracking_op.mp4"

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

input_prompt = "car. "
graph = bmf.graph()

object_tracking = bmf.create_module('object_tracking', {
    "grounding_dino_model_id": GROUNDING_DINO_MODEL_ID,
    "sam2_model_path": SAM2_MODEL_PATH,
    "sam2_config_path": SAM2_CONFIG_PATH,
    "text_prompt": input_prompt
})

video_stream = graph.decode({'input_path': input_video_path})

mask_stream = video_stream['video'].module('object_tracking',
                                            pre_module=object_tracking, scheduler=1)

mask_stream[0].encode(None, {"output_path": output_video_path}).run()

