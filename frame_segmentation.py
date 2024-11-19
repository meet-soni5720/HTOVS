import numpy as np
import bmf.hml.hmp as mp
import time
import torch
from bmf import Module, Log, LogLevel, ProcessResult, Packet, Timestamp, VideoFrame
from bmf.lib._bmf import sdk
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision.draw.color import ColorPalette
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from PIL import Image
from collections import deque
import json

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

class frame_segmentation(Module):

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
        self.sam2_model = build_sam2(self.sam2_config_path_, self.sam2_model_path_, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        #Initializing Grounding Dino model
        # self.grounding_model = load_model(
        #     model_config_path=self.grounding_dino_config_path_, 
        #     model_checkpoint_path=self.grounding_dino_model_path_,
        #     device=self.device
        # )
        self.processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id).to(self.device)

        self.frame_list_ = deque()
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25
        self.box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.result_arr = []

        Log.log(LogLevel.ERROR, "Load model takes", (time.time() - start_time))


    def load_image(self, image):
        # print(f"image shape: {image.shape}")
        img = Image.fromarray(image.astype('uint8'), "RGB")

        # transform = T.Compose(
        #     [
        #         T.RandomResize([800], max_size=1333),
        #         T.ToTensor(),
        #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )

        # image_transformed, _ = transform(img, None)
        inputs = self.processor(images=image, text=self.text_prompt_, return_tensors="pt").to(self.device)
        return img, inputs

    # def reset(self):
    #     # clear status
    #     self.eof_received_ = False
    #     while not self.frame_cache_.empty():
    #         self.frame_cache_.get()

    def visualize_mask(self, image, input_boxes, masks, class_ids, labels):
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        annotated_frame = self.box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = self.mask_annotator.annotate(scene=annotated_frame, detections=detections)

        return annotated_frame
    
    def process_segmentation_result(self, frame, input_boxes, masks):
        result = {}
        return frame, result

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

        # detect processing
        while self.frame_list_ or self.eof_received_:
            in_frame = self.frame_list_[0]
            del self.frame_list_[0]
            # print(f"frame data: {in_frame.cpu().frame().data()}")

            dst_md = sdk.MediaDesc().pixel_format(mp.kPF_RGB24)
            np_vf = sdk.bmf_convert(in_frame, sdk.MediaDesc(), dst_md).frame().plane(0).numpy()

            image_source, inputs = self.load_image(np_vf)
            self.sam2_predictor.set_image(image_source)
            # boxes, confidences, labels = predict(
            #     model=self.grounding_model,
            #     image=transformed_image,
            #     caption=self.text_prompt_,
            #     box_threshold=self.BOX_THRESHOLD,
            #     text_threshold=self.TEXT_THRESHOLD,
            # )
            print(image_source.size[::-1])
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

                image_source = np.array(image_source)
                annotated_frame = self.visualize_mask(image_source, input_boxes, masks, class_ids, labels)
                # rgbinfo = mp.PixelInfo(mp.PixelFormat.kPF_RGB24,
                #                         in_frame.frame().pix_info().space,
                #                         in_frame.frame().pix_info().range)
                annotated_frame, result = self.process_segmentation_result(annotated_frame, input_boxes, masks)
                self.result_arr.append(result)
            else:
                annotated_frame = image_source

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
        
            if len(self.frame_list_) == 0:
                break

        # add eof packet to output
        if self.eof_received_:
            for key in task.get_outputs():
                task.get_outputs()[key].put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            print(f"total time taken to complete the process: {time.time() - startProcess}")
            task.set_timestamp(Timestamp.DONE)

        with open('output.json', 'a') as json_file:
            json.dump(self.result_arr, json_file, indent=4)

        return ProcessResult.OK