import os
import cv2
import torch
import numpy as np
import supervision as sv
import argparse
import shutil
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy


VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grounded SAM2 continuous-id tracking on all images in a directory."
    )
    parser.add_argument("--image_dir", default="notebooks/videos/car", help="Directory containing input images.")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save masks/json/visualizations.")
    parser.add_argument("--text", default="car.", help="Grounding text prompt, e.g. 'car.'")
    parser.add_argument("--step", type=int, default=20, help="Run Grounding DINO every N images.")
    parser.add_argument("--save_video", action="store_true", help="Also save an mp4 visualization video.")
    parser.add_argument("--output_video_path", default="./outputs/output.mp4", help="Output video path when --save_video is set.")
    parser.add_argument("--propainter_mask_dir", default="propainter_masks", help="Subdirectory to save ProPainter-compatible binary PNG masks.")
    return parser.parse_args()


def list_images(image_dir):
    image_names = [
        p for p in os.listdir(image_dir)
        if os.path.splitext(p)[-1] in VALID_IMAGE_EXTENSIONS
    ]
    image_names.sort()
    if not image_names:
        raise RuntimeError(f"No images found in {image_dir}")
    return image_names


def save_propainter_mask(mask_array, output_path):
    propainter_mask = (mask_array > 0).astype(np.uint8) * 255
    cv2.imwrite(output_path, propainter_mask)


def get_aligned_frame_stem(frame_idx, name_width):
    return f"{frame_idx:0{name_width}d}"


def get_propainter_mask_name(frame_idx, name_width):
    return f"{get_aligned_frame_stem(frame_idx, name_width)}.png"


def save_empty_propainter_masks(frame_indices, mask_height, mask_width, output_dir, name_width):
    empty_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    for frame_idx in frame_indices:
        cv2.imwrite(os.path.join(output_dir, get_propainter_mask_name(frame_idx, name_width)), empty_mask)


def prepare_sam2_jpeg_frames(image_dir, image_names, output_dir, name_width):
    """
    SAM2 video predictor only accepts JPEG frames named like 0.jpg, 1.jpg, ...
    Convert/copy arbitrary input images to such a frame directory while keeping
    image_names as the mapping back to original filenames.
    """
    frame_dir = os.path.join(output_dir, "sam2_jpeg_frames")
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    CommonUtils.creat_dirs(frame_dir)
    for idx, image_name in enumerate(image_names):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")
        frame_name = f"{get_aligned_frame_stem(idx, name_width)}.jpg"
        cv2.imwrite(os.path.join(frame_dir, frame_name), image)
    return frame_dir

# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = args.text

# `image_dir` can contain jpg/png/bmp images with arbitrary filenames.
image_dir = args.image_dir
# 'output_dir' is the directory to save the annotated frames
output_dir = args.output_dir
# 'output_video_path' is the path to save the final video
output_video_path = args.output_video_path
# create the output directory
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
propainter_mask_dir = os.path.join(output_dir, args.propainter_mask_dir)
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
if os.path.exists(propainter_mask_dir):
    shutil.rmtree(propainter_mask_dir)
CommonUtils.creat_dirs(propainter_mask_dir)
# scan all image names in this directory and prepare SAM2-compatible JPEG frames
frame_names = list_images(image_dir)
name_width = max(5, len(str(len(frame_names) - 1)))
video_dir = prepare_sam2_jpeg_frames(image_dir, frame_names, output_dir, name_width)

# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir)
step = args.step # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0
frame_object_count = {}
"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    # continue
    img_path = os.path.join(image_dir, frame_names[start_frame_idx])
    image = Image.open(img_path).convert("RGB")
    image_base_name = os.path.splitext(frame_names[start_frame_idx])[0]
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
    mask_dict.mask_height = image.height
    mask_dict.mask_width = image.width

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"] # .cpu().numpy()
    # print("results[0]",results[0])
    OBJECTS = results[0]["labels"]
    if input_boxes.shape[0] != 0:

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        """
        Step 3: Register each object's positive points to video predictor
        """

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")
    else:
        print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
        if len(sam2_masks.labels) != 0:
            mask_dict = sam2_masks

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
    frame_object_count[start_frame_idx] = objects_count
    print("objects_count", objects_count)
    
    if len(mask_dict.labels) == 0:
        empty_frame_names = frame_names[start_frame_idx:start_frame_idx+step]
        empty_frame_indices = range(start_frame_idx, start_frame_idx + len(empty_frame_names))
        mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list=empty_frame_names)
        save_empty_propainter_masks(empty_frame_indices, mask_dict.mask_height, mask_dict.mask_width, propainter_mask_dir, name_width)
        print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
        continue
    else:
        video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )
        
        video_segments = {}  # output the following {step} frames tracking masks
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id))
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = os.path.splitext(frame_names[out_frame_idx])[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        print("video_segments:", len(video_segments))
    """
    Step 5: save the tracking masks and json files
    """
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)
        propainter_mask_name = get_propainter_mask_name(frame_idx, name_width)
        save_propainter_mask(mask_img, os.path.join(propainter_mask_dir, propainter_mask_name))

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        frame_masks_info.to_json(json_data_path)
       

CommonUtils.draw_masks_and_box_with_supervision(image_dir, mask_data_dir, json_data_dir, result_dir)

print("try reverse tracking")
start_object_id = 0
object_info_dict = {}
for frame_idx, current_object_count in frame_object_count.items():
    print("reverse tracking frame", frame_idx, frame_names[frame_idx])
    new_object_ids = list(range(start_object_id + 1, current_object_count + 1))
    if frame_idx == 0 or len(new_object_ids) == 0:
        start_object_id = current_object_count
        continue

    video_predictor.reset_state(inference_state)
    image_base_name = os.path.splitext(frame_names[frame_idx])[0]
    json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
    json_data = MaskDictionaryModel().from_json(json_data_path)
    mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
    mask_array = np.load(mask_data_path)
    added_reverse_prompt = False
    for object_id in new_object_ids:
        if object_id not in json_data.labels:
            continue
        object_mask = mask_array == object_id
        if object_mask.sum() == 0:
            continue
        print("reverse tracking object", object_id)
        object_info_dict[object_id] = json_data.labels[object_id]
        video_predictor.add_new_mask(inference_state, frame_idx, object_id, object_mask)
        added_reverse_prompt = True
    start_object_id = current_object_count
    if not added_reverse_prompt:
        continue

    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2,  start_frame_idx=frame_idx, reverse=True):
        image_base_name = os.path.splitext(frame_names[out_frame_idx])[0]
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        json_data = MaskDictionaryModel().from_json(json_data_path)
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
        mask_array = np.load(mask_data_path)
        # merge the reverse tracking masks with the original masks
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu()
            if out_mask.sum() == 0:
                print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                continue
            object_info = object_info_dict[out_obj_id]
            object_info.mask = out_mask[0]
            object_info.update_box()
            json_data.labels[out_obj_id] = object_info
            mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
            mask_array[object_info.mask] = out_obj_id
        
        np.save(mask_data_path, mask_array)
        save_propainter_mask(mask_array, os.path.join(propainter_mask_dir, get_propainter_mask_name(out_frame_idx, name_width)))
        json_data.to_json(json_data_path)

        



"""
Step 6: Draw the results and save the video
"""
CommonUtils.draw_masks_and_box_with_supervision(image_dir, mask_data_dir, json_data_dir, result_dir+"_reverse")

if args.save_video:
    create_video_from_images(result_dir, output_video_path, frame_rate=15)
