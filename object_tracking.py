import os
import numpy as np
import bmf.hml.hmp as mp
import time
import torch
from bmf import Module, Log, LogLevel, ProcessResult, Packet, Timestamp, VideoFrame
from bmf.lib._bmf import sdk
import supervision as sv
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision.draw.color import ColorPalette
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from PIL import Image
from collections import deque
import json
import cv2
from gsam2.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from gsam2.utils.common_utils import CommonUtils
import copy

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

class object_tracking(Module):

    def __init__(self, node=None, option=None):
        self.node_ = node
        self.option_ = option
        self.eof_received_ = False

        if option is None:
            Log.log(LogLevel.ERROR, "Option is none")
            return

        if 'grounding_dino_model_id' in option.keys():
            self.grounding_dino_model_id = option['grounding_dino_model_id']
        else:
            Log.log(LogLevel.ERROR, "Grounding dino model id not provided!")
            return
        
        if 'sam2_model_path' in option.keys():
            self.sam2_model_path_ = option['sam2_model_path']
        else:
            Log.log(LogLevel.ERROR, "Sam2 model path not provided!")
            return
        
        if 'sam2_config_path' in option.keys():
            self.sam2_config_path_ = option['sam2_config_path']
        else:
            Log.log(LogLevel.ERROR, "Sam2 config path not provided!")
            return
        
        if 'text_prompt' in option.keys():
            self.text_prompt_ = option['text_prompt']
        else:
            Log.log(LogLevel.ERROR, "text prompt not provided!")
            return
        
        start_time = time.time()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)

        #Initializing SAM2 model
        self.video_predictor = build_sam2_video_predictor(self.sam2_config_path_, self.sam2_model_path_)
        self.sam2_model = build_sam2(self.sam2_config_path_, self.sam2_model_path_, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id).to(self.device)

        self.frame_list_ = deque()
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
        self.box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.result_arr = []
        self.frame_counter_ = 0
        self.frame_counter_queue_ = deque()
        self.output_dir_ = "video_tracking_output"
        self.mask_data_dir_ = os.path.join(self.output_dir_, "mask_data")
        self.json_data_dir_ = os.path.join(self.output_dir_, "json_data")
        self.result_dir_ = os.path.join(self.output_dir_, "result")

        CommonUtils.creat_dirs(self.mask_data_dir_)
        CommonUtils.creat_dirs(self.json_data_dir_)

        if not os.path.exists("frames"):
            os.makedirs("frames")

        Log.log(LogLevel.ERROR, "Load model takes", (time.time() - start_time))


    def load_image(self, image):
        # print(f"image shape: {image.shape}")
        img = Image.fromarray(image.astype('uint8'), "RGB")
        inputs = self.processor(images=image, text=self.text_prompt_, return_tensors="pt").to(self.device)
        return img, inputs
    
    def video_segment(self, video_dir):
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]

        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        print(frame_names)
        inference_state = self.video_predictor.init_state(video_path=video_dir)
        step = 10 # the step to sample frames for Grounding DINO predictor
        PROMPT_TYPE_FOR_VIDEO = "mask"
        objects_count = 0
        sam2_masks = MaskDictionaryModel()

        for start_frame_idx in range(0, len(frame_names), step):
            img_path = os.path.join(video_dir, frame_names[start_frame_idx])
            image = Image.open(img_path)
            image_base_name = frame_names[start_frame_idx].split(".")[0]
            mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

            image_source, inputs = self.load_image(np.array(image))

            with torch.no_grad():
                outputs = self.grounding_model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=self.BOX_THRESHOLD,
                        text_threshold=self.TEXT_THRESHOLD,
                        target_sizes=[image_source.size[::-1]]
                    )
            
            input_boxes = results[0]["boxes"].cpu().numpy()
            labels = results[0]["labels"]
            confidences = results[0]["scores"].cpu()
            
            if(len(input_boxes) > 0):
                self.sam2_predictor.set_image(image_source)
                #SAM2 predictions
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )

                if masks.ndim == 4:
                    masks = masks.squeeze(1)


                confidences = confidences.numpy().tolist()
                class_names = labels

                class_ids = np.array(list(range(len(class_names))))

                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(class_names, confidences)
                ]

                if mask_dict.promote_type == "mask":
                    mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(self.device), box_list=torch.tensor(input_boxes), label_list=labels)
                else:
                    raise NotImplementedError("SAM 2 video predictor only support mask prompts")

                objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.4, objects_count=objects_count)
            else:
                print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
                mask_dict = sam2_masks

            #Propogating in video
            if len(mask_dict.labels) == 0:
                mask_dict.save_empty_mask_and_json(self.mask_data_dir_, self.json_data_dir_, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
                print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                continue
            else:
                print("tracking")
                self.video_predictor.reset_state(inference_state)

                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                            inference_state,
                            start_frame_idx,
                            object_id,
                            object_info.mask,
                        )
                
                video_segments = {}  # output the following {step} frames tracking masks
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                    frame_masks = MaskDictionaryModel()
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0)
                        object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info
                        image_base_name = frame_names[out_frame_idx].split(".")[0]
                        frame_masks.mask_name = f"mask_{image_base_name}.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]

                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = copy.deepcopy(frame_masks)

            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id

                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(self.mask_data_dir_, frame_masks_info.mask_name), mask_img)

                json_data = frame_masks_info.to_dict()
                json_data_path = os.path.join(self.json_data_dir_, frame_masks_info.mask_name.replace(".npy", ".json"))
                with open(json_data_path, "w") as f:
                    json.dump(json_data, f)
            
        CommonUtils.draw_masks_and_box_with_supervision(video_dir, self.mask_data_dir_, self.json_data_dir_, self.result_dir_)


    def process(self, task):
        startProcess = time.time()
        image_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        while not image_queue.empty():
            pkt = image_queue.get()
            if pkt.timestamp == Timestamp.EOF:
                self.eof_received_ = True
                break
            if pkt.is_(VideoFrame):
                self.frame_list_.append(pkt.get(VideoFrame))
                self.frame_counter_queue_.append(self.frame_counter_)
                self.frame_counter_+=1

        # Save images in a directory: requirement for SAM2 model for object tracking
        while self.frame_list_ or self.eof_received_:
            in_frame = self.frame_list_[0]
            self.frame_list_.popleft()

            frame_no = self.frame_counter_queue_[0]
            self.frame_counter_queue_.popleft()

            dst_md = sdk.MediaDesc().pixel_format(mp.kPF_RGB24)
            np_vf = sdk.bmf_convert(in_frame, sdk.MediaDesc(), dst_md).frame().plane(0).numpy()

            frame_name = f"./frames/{frame_no}.jpg"
            np_vf = Image.fromarray(np_vf).convert("RGB")
            np_vf.save(frame_name)
        
            if len(self.frame_list_) == 0:
                break

        # add eof packet to output
        if self.eof_received_:
            self.video_segment("./frames")

            image_files = [f for f in os.listdir(self.result_dir_)]
            image_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
            print(image_files)

            for f in image_files:
                image_path = os.path.join(self.result_dir_, f)
                annotated_frame = Image.open(image_path).convert("RGB")

                rgbinfo = mp.PixelInfo(mp.PixelFormat.kPF_RGB24)
                
                annotated_frame = np.array(annotated_frame)
                np_op = mp.from_numpy(annotated_frame)

                out_f = mp.Frame(np_op, rgbinfo)
                out_vf = VideoFrame(out_f)
                out_vf.pts = in_frame.pts
                out_vf.time_base = in_frame.time_base
                out_pkt = Packet(out_vf)
                out_pkt.timestamp = out_vf.pts
                output_queue.put(out_pkt)
            
            for key in task.get_outputs():
                task.get_outputs()[key].put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            print(f"total time taken to complete the process: {time.time() - startProcess}")
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK