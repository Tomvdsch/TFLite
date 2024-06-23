'''
This script demonstrates the use of TFLite models for object detection and re-identification
using YOLO-World and OSNET-AIN models. It captures frames from a video or a YouTube stream,
detects objects, and re-identifies detected persons across frames.

Dependencies:
- OpenCV
- Torch
- TensorFlow
- Supervision
- SciPy
- torchvision
- PyTube
- TFLite Runtime

Usage:
python script_name.py [YOLO_World TFLite model path] [OSNET_AIN TFLite model path] [Text or JSON file for class names]
                      [Optional: Output directory] [Optional: Number of threads] [Optional: External delegate path]
                      [Optional: External delegate options]
'''
import os
import json
import argparse
import os.path as osp

import cv2
import tqdm
import torch
import time
import numpy as np
import tensorflow as tf
import supervision as sv
from scipy.spatial import distance
from torchvision.ops import nms
from pytube import YouTube
import tflite_runtime.interpreter as tflite

#Init supervision (To draw bbox + class, id and confidence score)
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)
'''
    Parse command line arguments.

    Returns:
        args : Parsed command line arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser('YOLO-World TFLite (INT8) Demo')
    parser.add_argument('yolo_world', 
                        default='./Files/yolo_world_x_coco_zeroshot_rep_integer_quant.tflite', 
                        help='YOLO_World TFLite Model `yolo_world.tflite`') #YOLO-World model
    parser.add_argument('osnet_ain', 
                        default='./Files/osnet_ain_x1_0_M_integer_quant.tflite',
                        help='OSNET_AIN TFLite Model `osnet_ain.tflite`') #OSNET-AIN model
    parser.add_argument('text',
                        default='./Files/test_class_texts.json',
                        help='detecting texts (str, txt, or json), should be consistent with the ONNX model') #Detection classes
    parser.add_argument('--output_dir',
                        default='./output',
                        help='directory to save output files') #Output directory
    parser.add_argument('--num_threads', 
                        default=None, 
                        type=int, 
                        help='number of threads') #Can be specified but usually not used
    parser.add_argument('-e', '--ext_delegate',
                        help='external_delegate_library path') #To use NPU use -e /usr/lib/libvx_delegate.so
    parser.add_argument('-o', '--ext_delegate_options',
                        help='external delegate options, format: "option1: value1; option2: value2"') #Can be specified but usually not used
    args = parser.parse_args()
    return args

#----------------------------------------------------------------------------------------------------------
#ADD:  score, nms and feature threshold to arguments so user can set these values (No need to hardcode them)
#----------------------------------------------------------------------------------------------------------

'''
Prepare image for YOLO-World model (Resize)

Args:
    image (numpy.ndarray): Input image
    size (tuple): Target size for resizing

Returns:
    tuple: Preprocessed image, scale factor, and padding parameters
'''
def preprocess(image, size=(640, 640)):
    h, w = image.shape[:2]
    max_size = max(h, w)
    scale_factor = size[0] / max_size
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image
    image = cv2.resize(pad_image, size,
                       interpolation=cv2.INTER_LINEAR).astype('float32')
    image /= 255.0
    image = image[None]
    return image, scale_factor, (pad_h, pad_w)

'''
    Generate anchors per level to scale bounding boxes correctly

    Args:
        feat_size (tuple): Feature map size
        stride (int): Stride value
        offset (float): Offset value

    Returns:
        anchors (torch.Tensor): Anchors
'''
def generate_anchors_per_level(feat_size, stride, offset=0.5):
    h, w = feat_size
    shift_x = (torch.arange(0, w) + offset) * stride
    shift_y = (torch.arange(0, h) + offset) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x)
    anchors = torch.stack([xx, yy]).reshape(2, -1).transpose(0, 1)
    return anchors

'''
    Generate anchors to scale bounding boxes correctly

    Args:
        feat_sizes (list): List of feature map sizes
        strides (list): List of stride values
        offset (float): Offset value

    Returns:
        anchors (torch.Tensor): Anchors
'''
def generate_anchors(feat_sizes=[(80, 80), (40, 40), (20, 20)],
                     strides=[8, 16, 32],
                     offset=0.5):
    anchors = [
        generate_anchors_per_level(fs, s, offset)
        for fs, s in zip(feat_sizes, strides)
    ]
    anchors = torch.cat(anchors)
    return anchors

'''
    Correct offsets in bounding box

    Args:
        points (torch.Tensor): Points tensor
        pred_bboxes (torch.Tensor): Predicted bounding boxes
        stride (torch.Tensor): Stride values

    Returns:
        bboxes (torch.Tensor): Decoded bounding boxes
'''
def simple_bbox_decode(points, pred_bboxes, stride):

    pred_bboxes = pred_bboxes * stride[None, :, None]
    x1 = points[..., 0] - pred_bboxes[..., 0]
    y1 = points[..., 1] - pred_bboxes[..., 1]
    x2 = points[..., 0] + pred_bboxes[..., 2]
    y2 = points[..., 1] + pred_bboxes[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return bboxes

'''
    Visualize bounding boxes, class, ID, and confidence score on frame

    Args:
        image (numpy.ndarray): Input image
        bboxes (numpy.ndarray): Bounding boxes
        labels (numpy.ndarray): Class labels
        scores (numpy.ndarray): Confidence scores
        texts (list): List of class names
        id (int): ID of detected object

    Returns:
        image (numpy.ndarray): Annotated image
'''
def visualize(image, bboxes, labels, scores, texts, id):
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {id} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image

'''
    Get YouTube stream URL

    Args:
        youtube_url (str): YouTube video URL

    Returns:
        stream.url (str): Stream URL
'''
#Function to use if ML needs to be ran on a youtube video (Used for testing)
def get_youtube_stream_url(youtube_url):
    yt = YouTube(youtube_url)
    #Get the first stream with progressive download and mp4 format
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    return stream.url    

'''
    Prepare image for OSNET-AIN model.

    Args:
        input_details_FE (list): Input details of OSNET-AIN model
        output_details_FE (list): Output details of OSNET-AIN model
        image (numpy.ndarray): Input image

    Returns:
        image (numpy.ndarray): Preprocessed image
'''
def preprocess_feature_extraction(input_details_FE, output_details_FE, image):
    image = cv2.resize(image, (input_details_FE[0]['shape'][2], input_details_FE[0]['shape'][1]))
    image = np.expand_dims(image, axis=0).astype('float32')
    image = (image - 127.5) / 127.5
    return image

'''
    Run OSNET-AIN model and return features of detected person

    Args:
        image (numpy.ndarray): Input image.
        interpreter_FE (tflite.Interpreter): OSNET-AIN interpreter
        input_details_FE (list): Input details of OSNET-AIN model
        output_details_FE (list): Output details of OSNET-AIN model

    Returns:
        features (numpy.ndarray): Extracted features
        deltaOSNET (float): Inference time for the OSNET-AIN model (0 if no person detected)
'''
def extract(image, interpreter_FE, input_details_FE, output_details_FE):
    image = preprocess_feature_extraction(input_details_FE, output_details_FE, image)
    interpreter_FE.set_tensor(input_details_FE[0]['index'], image)
    startTime = time.time()
    interpreter_FE.invoke()
    deltaOSNET = time.time() - startTime
    print("Inference time:", '%.1f' % (deltaOSNET * 1000), "ms\n")
    features = interpreter_FE.get_tensor(output_details_FE[0]['index'])
    return features, deltaOSNET

'''
    Perform inference for object detection and re-identification on a single image

    This function processes an input image, performs object detection using a YOLO model,
    extracts features using an OSNET-AIN model if a person is detected, and optionally
    visualizes and saves the results

    Args:
        interp (tflite.Interpreter): YOLO-World interpreter
        input_details (list): Input details of YOLO-World model
        output_details (list): Output details of YOLO-World model
        interp_FE (tflite.Interpreter): OSNET-AIN interpreter
        input_details_FE (list): Input details of OSNET-AIN model
        output_details_FE (list): Output details of OSNET-AIN model
        image_path (str): Path to the input image
        image_out (str): Filename for the output image
        texts (list): List of class names
        priors (torch.Tensor): Anchors used for bounding box decoding
        strides (torch.Tensor): Strides used for bounding box decoding
        output_dir (str): Directory to save output images
        size (tuple): Target size for resizing the image (default is (640, 640))
        vis (bool): Whether to visualize and save the results (default is False)
        score_thr (float): Score threshold for filtering detections (default is 0.5)
        nms_thr (float): Non-maximum suppression threshold (default is 0.5)
        max_dets (int): Maximum number of detections to retain (default is 1)
        cam_id (int): Camera ID for the input image (default is 0)
        ftr_thr (float): Feature similarity threshold for re-identification (default is 0.5)
        dt_prs (dict): Dictionary to store detected persons data (default is empty dict)
        id (int): Initial ID for detected persons (default is 0)

    Returns:
        tuple: A tuple containing:
            - dt_prs (dict): Updated dictionary with detected persons data
            - id (int): Updated ID for detected persons
            - deltaYOLO (float): Inference time for the YOLO model
            - deltaOSNET (float): Inference time for the OSNET-AIN model (0 if no person detected)
'''
def inference_per_sample(interp,
                         input_details,
                         output_details,
                         interp_FE,
                         input_details_FE,
                         output_details_FE,
                         image_path,
                         image_out,
                         texts,
                         priors,
                         strides,
                         output_dir,
                         size=(640, 640),
                         vis=False,
                         score_thr=0.5,
                         nms_thr=0.5,
                         max_dets=1,
                         cam_id=0,
                         ftr_thr=0.5,
                         dt_prs={},
                         id=0,):

    #Save original image + shape (So processed image can be saved in same format)
    ori_image = image_path
    h, w = ori_image.shape[:2]

    #Prepare image for YOLO-World model
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)

    # inference
    interp.set_tensor(input_details[0]['index'], image)

    #Run YOLO-World model
    startTime = time.time()
    interp.invoke()
    deltaYOLO = time.time() - startTime
    print("Inference time:", '%.1f' % (deltaYOLO * 1000), "ms\n")

    ''' #RUN YOLO 400 Times
    for i in range(400):
        startTime = time.time()
        interp.invoke()
        deltaYOLO = time.time() - startTime
        print("Inference time:", '%.1f' % (deltaYOLO * 1000), "ms\n")
    '''

    #Get scores and bboxes detected by YOLO-World model
    scores = interp.get_tensor(output_details[1]['index'])
    bboxes = interp.get_tensor(output_details[0]['index'])

    #Save the original detected scores and bboxes
    ori_scores = torch.from_numpy(scores[0])
    ori_bboxes = torch.from_numpy(bboxes)

    #Decode bbox cordinates 
    decoded_bboxes = simple_bbox_decode(priors, ori_bboxes, strides)[0]

    #Initialize scores, labels, and bbox lists
    scores_list = []
    labels_list = []
    bboxes_list = []

    #Loop through all detection classes
    for cls_id in range(len(texts)):
        #Extract scores for the current class
        cls_scores = ori_scores[:, cls_id]
        
        #Create a array of labels for the current class
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        
        #Apply NMS to filter out overlapping bboxes
        keep_idxs = nms(decoded_bboxes, cls_scores, iou_threshold=0.5)
        
        #Select the kept bboxes and scores
        cur_bboxes = decoded_bboxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        
        #Save the filtered bboxes, scores and labels
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)

    #Concatenate the lists of scores, labels and bboxes
    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    #Apply the detection threshold to filter out low confidence detections
    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]

    #Apply NMS again to further remove redundant boxes
    keep_idxs = nms(bboxes, scores, iou_threshold=nms_thr)

    #Limit the number of detections
    num_dets = min(len(keep_idxs), max_dets)
    bboxes = bboxes[keep_idxs].unsqueeze(0)
    scores = scores[keep_idxs].unsqueeze(0)
    labels = labels[keep_idxs].unsqueeze(0)

    #Convert tensors to numpy arrays and select the top detections
    scores = scores[0, :num_dets].numpy()
    bboxes = bboxes[0, :num_dets].numpy()
    labels = labels[0, :num_dets].numpy()

    #Adjust bounding boxes based on padding and scaling factors
    bboxes -= np.array([pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor

    #Clip bounding box coordinates to be within the image dimensions
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)

    #Set deltaOSNET to 0 in case no people were detected
    deltaOSNET = 0

    if vis:
        #If a person is detected run OSNET-AIN model
        if(num_dets > 0):
            x1, y1, x2, y2 = bboxes[0]

            #Crop image to detected person bbox
            cropped_image = ori_image[round(y1):round(y2), round(x1):round(x2)]

            #Run OSNET-AIN model
            extracted_features, deltaOSNET = extract(cropped_image, interp_FE, input_details_FE, output_details_FE)

            #If features were extracted correctly
            if len(extracted_features) != 0:
                #Add new person if data is empty
                if not dt_prs:
                    dt_prs[f"id_{id}"] = {
                        "extracted_features": extracted_features,
                        "id": id,
                        "camera_id": cam_id,
                        "class_name": labels[0],
                        "bbox": bboxes[0],
                        "confidence": scores[0],
                        "identification_color": np.random.randint(0, 255, size=3),
                        }
                    id += 1
                else:
                   #Search Top 1 identification_score person identification
                    best_match = np.array(
                        [
                            {
                                "id": value["id"],
                                "class_name": value["class_name"],
                                "identification_color": value["identification_color"],
                                "identification_score": distance.cosine(
                                    np.ravel(np.mean(value["extracted_features"], axis=0)),
                                    np.ravel(extracted_features),
                                ),
                            }
                            for value in dt_prs.values()
                        ]
                    )

                    best_match = sorted(best_match, key=lambda d: d["identification_score"], reverse=False)[0]

                    #Add or replace data for new person profile if feature simularity is above the set threshold
                    if best_match["identification_score"] < ftr_thr: #Lower identification_score = more similar
                        dt_prs[f"id_{best_match['id']}"] = {
                            "extracted_features": np.vstack(
                                (
                                    dt_prs[f"id_{best_match['id']}"][
                                    "extracted_features"
                                    ],
                                    extracted_features,
                                )
                            )
                            if dt_prs[f"id_{best_match['id']}"][
                                "extracted_features"
                            ].shape[0]
                            < 512
                            else np.vstack(
                                (
                                    extracted_features,
                                    dt_prs[f"id_{best_match['id']}"][
                                        "extracted_features"
                                    ][1:],
                                )
                            ),
                            "id": best_match["id"],
                            "camera_id": cam_id,
                            "class_name": best_match["class_name"],
                            "bbox": bboxes[0],
                            "confidence": scores[0],
                            "identification_color": best_match["identification_color"],
                        }
                        image_out = visualize(ori_image, bboxes, labels, scores, texts, best_match["id"])
                    else:
                        dt_prs[f"id_{id}"] = {
                            "extracted_features": extracted_features,
                            "id": id,
                            "camera_id": cam_id,
                            "class_name": labels[0],
                            "bbox": bboxes[0],
                            "confidence": scores[0],
                            "identification_color": np.random.randint(0, 255, size=3),
                        }
                        image_out = visualize(ori_image, bboxes, labels, scores, texts, id)
                        id += 1
        #Write image to output directoy
        cv2.imwrite(osp.join(output_dir, "image.jpg"), image_out)
        print(f"detecting {num_dets} objects.")
        return dt_prs, id, deltaYOLO, deltaOSNET
    else:
        return dt_prs, id, deltaYOLO, deltaOSNET

def main():
    #Load in parameters
    args = parse_args()

    #Init variables
    ext_delegate = None
    ext_delegate_options = {}

    #Parse extenal delegate options
    if args.ext_delegate_options is not None:
        options = args.ext_delegate_options.split(';')
        for o in options:
            kv = o.split(':')
            if (len(kv) == 2):
                ext_delegate_options[kv[0].strip()] = kv[1].strip()
            else:
                raise RuntimeError('Error parsing delegate option: ' + o)

    #Load external delegate (Used for NPU)
    if args.ext_delegate is not None:
        print('Loading external delegate from {} with args: {}'.format(args.ext_delegate, ext_delegate_options))
        ext_delegate = [tflite.load_delegate(args.ext_delegate, ext_delegate_options)]

    #Get models from parameters
    yolo_world_tflite_file = args.yolo_world
    osnet_ain_tflite_file = args.osnet_ain

    #Init YOLO-World model
    interpreter = tflite.Interpreter(model_path=yolo_world_tflite_file,
                                      experimental_preserve_all_tensors=True,
                                      experimental_delegates=ext_delegate,
                                      num_threads=args.num_threads)
    
    #Set input / output details for YOLO-World model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    try:
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Failed to allocate tensors: {e}")
        return

    #Init OSNET-AIN model
    interpreter_FE = tflite.Interpreter(model_path=osnet_ain_tflite_file,
                                          experimental_preserve_all_tensors=True,
                                          experimental_delegates=ext_delegate,
                                          num_threads=args.num_threads)
    
    #Set input / output details for YOLO-World model
    input_details_FE = interpreter_FE.get_input_details()
    output_details_FE = interpreter_FE.get_output_details()
    try:
        interpreter_FE.allocate_tensors()
    except Exception as e:
        print(f"Failed to allocate tensors: {e}")
        return

    print("Init TFLite Interpter")

    #Set output directory for results
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    #Init cams + videos
    cam = {}
    videos = np.array(["https://youtu.be/Tn70NxIMk2Q"])

    #Enable of a youtube video is used
    youtubeVideos = True

    #Load videos
    total_cam = len(videos)
    for i in range(total_cam):
        if youtubeVideos == True:
            stream_url = get_youtube_stream_url(videos[i])
            cam[f"cam_{i}"] = cv2.VideoCapture(stream_url)
        else:
            cam[f"cam_{i}"] = cv2.VideoCapture(videos[i])

    #Load detection classes
    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines]
    elif args.text.endswith('.json'):
        texts = json.load(open(args.text))
    else:
        texts = [[t.strip()] for t in args.text.split(',')]

    #Init variables
    size = (640, 640)
    strides = [8, 16, 32]

    #Prepare anchors, since TFLite models do not contain anchors, due to INT8 quantization.
    featmap_sizes = [(size[0] // s, size[1] // s) for s in strides]
    flatten_priors = generate_anchors(featmap_sizes, strides=strides)
    mlvl_strides = [
        flatten_priors.new_full((featmap_size[0] * featmap_size[1] * 1, ),
                                stride)
        for featmap_size, stride in zip(featmap_sizes, strides)
    ]
    flatten_strides = torch.cat(mlvl_strides)

    #Variable for save detected person
    detected_persons = {}
    id = 0
    counter = 1

    with open("benchmark.txt", "a") as file:

        while(True):
            images = {}
            # Get frames one by one
            for i in range(total_cam):
                ret, images[f"image_{i}"] = cam[f"cam_{i}"].read()
                if not ret:
                    print(f"Error reading frame from camera {i}")
                    continue

            #Run models frame by frame
            for i in range(total_cam):
                print("Start to inference.")
                print("Frame counter:")
                print(counter)
                detected_persons, id, deltaYOLO, deltaOSNET = inference_per_sample(interpreter,
                                                                                    input_details,
                                                                                    output_details,
                                                                                    interpreter_FE,
                                                                                    input_details_FE,
                                                                                    output_details_FE,
                                                                                    images[f"image_{i}"],
                                                                                    images[f"image_{i}"],
                                                                                    texts,
                                                                                    flatten_priors[None],
                                                                                    flatten_strides,
                                                                                    output_dir=output_dir,
                                                                                    vis=True,
                                                                                    score_thr=0.3,
                                                                                    nms_thr=0.5,
                                                                                    cam_id=i,
                                                                                    ftr_thr=0.4,
                                                                                    dt_prs=detected_persons,
                                                                                    id=id)
                file.write(f"({counter}, {deltaYOLO}, {deltaOSNET})\n")
                counter += 1


if __name__ == "__main__":
    main()
