from frame_segmentation import frame_segmentation
import cv2
import numpy as np

class person_face_blur(frame_segmentation):
    def __init__(self, node=None, option=None):
        super().__init__(node, option)
    

    def process_segmentation_result(self, frame, input_boxes, masks):
        kernel_size = (35, 35)
        combined_mask = np.any(masks > 0, axis=0).astype(np.uint8) * 255
        frame_copy = frame.copy()
        blurred = cv2.GaussianBlur(frame_copy, kernel_size, 0)

        mask_3ch = np.stack([combined_mask] * 3, axis=-1)
        blurred_faces_frame = np.where(mask_3ch == 255, blurred, frame)
        result = {
            "face_detected": len(input_boxes)
        }

        return blurred_faces_frame, result