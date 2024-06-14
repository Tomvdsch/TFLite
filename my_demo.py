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


def parse_args():
    parser = argparse.ArgumentParser('YOLO-World TFLite (INT8) Demo')
    parser.add_argument('yolo_world', help='YOLO_World TFLite Model `yolo_world.tflite`')
    parser.add_argument('osnet_ain', help='OSNET_AIN TFLite Model `osnet_ain.tflite`')
    parser.add_argument(
        'text',
        help=
        'detecting texts (str, txt, or json), should be consistent with the ONNX model'
    )
    parser.add_argument('--output-dir',
                        default='./output',
                        help='directory to save output files')
    parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument(
      '-e', '--ext_delegate', help='external_delegate_library path')
    parser.add_argument(
      '-o',
      '--ext_delegate_options',
      help='external delegate options, \
            format: "option1: value1; option2: value2"')
    args = parser.parse_args()
    return args


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


def generate_anchors_per_level(feat_size, stride, offset=0.5):
    h, w = feat_size
    shift_x = (torch.arange(0, w) + offset) * stride
    shift_y = (torch.arange(0, h) + offset) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x)
    anchors = torch.stack([xx, yy]).reshape(2, -1).transpose(0, 1)
    return anchors


def generate_anchors(feat_sizes=[(80, 80), (40, 40), (20, 20)],
                     strides=[8, 16, 32],
                     offset=0.5):
    anchors = [
        generate_anchors_per_level(fs, s, offset)
        for fs, s in zip(feat_sizes, strides)
    ]
    anchors = torch.cat(anchors)
    return anchors


def simple_bbox_decode(points, pred_bboxes, stride):

    pred_bboxes = pred_bboxes * stride[None, :, None]
    x1 = points[..., 0] - pred_bboxes[..., 0]
    y1 = points[..., 1] - pred_bboxes[..., 1]
    x2 = points[..., 0] + pred_bboxes[..., 2]
    y2 = points[..., 1] + pred_bboxes[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return bboxes


def visualize(image, bboxes, labels, scores, texts, id):
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {id} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image

# Function to use if ML needs to be ran on a youtube video
def get_youtube_stream_url(youtube_url):
    yt = YouTube(youtube_url)
    # Get the first stream with progressive download and mp4 format
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    return stream.url

def init_feature_extraction(osnet_ain_model):
    print("Loading configuration for Feature Extraction...")

    #model = r"C:\Users\Tom.van.der.Schaaf\Desktop\Code\YOLO-World\deploy\Files\osnet_ain_x1_0_M_integer_quant.tflite"
    model = osnet_ain_model
        
    print("Loading TFLite model for Feature Extraction...")
    interpreter_FE = tflite.Interpreter(model_path=model,
                                         experimental_preserve_all_tensors=True)
    print("Feature Extraction model setup.")
    try: 
        interpreter_FE.allocate_tensors()
    except Exception as e:
        print(f"Failed to allocate tensors: {e}")
        return

    input_details_FE = interpreter_FE.get_input_details()
    output_details_FE = interpreter_FE.get_output_details()

    print(input_details_FE[0]['index'])
    print(output_details_FE[0]['index'])
        
    print("Feature Extraction model loaded successfully.")

    return input_details_FE, output_details_FE, interpreter_FE
    
def preprocess_feature_extraction(input_details_FE, output_details_FE, image):
    print("Preprocessing image for feature extraction...")
    image = cv2.resize(image, (input_details_FE[0]['shape'][2], input_details_FE[0]['shape'][1]))
    image = np.expand_dims(image, axis=0).astype('float32')
    image = (image - 127.5) / 127.5
    print("Image preprocessing for feature extraction completed.")
    return image
    
def extract(image, osnet_ain_model):
    print("Running feature extraction...")
    input_details_FE, output_details_FE, interpreter_FE = init_feature_extraction(osnet_ain_model)
    image = preprocess_feature_extraction(input_details_FE, output_details_FE, image)
    interpreter_FE.set_tensor(input_details_FE[0]['index'], image)
    interpreter_FE.invoke()
        
    features = interpreter_FE.get_tensor(output_details_FE[0]['index'])
    print("Feature extraction completed.")
    return features


def inference_per_sample(interp,
                         osnet_ain_model,
                         image_path,
                         image_out,
                         texts,
                         priors,
                         strides,
                         output_dir,
                         size=(640, 640),
                         vis=False,
                         score_thr=0.05,
                         nms_thr=0.3,
                         max_dets=1,
                         cam_id=0,
                         ftr_thr=0.4,
                         dt_prs={},
                         id=0,):

    # input / output details from TFLite
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # load image from path
    ori_image = image_path
    h, w = ori_image.shape[:2]
    #print(h, w)
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)

    # inference
    interp.set_tensor(input_details[0]['index'], image)
    
    startTime = time.time()
    interp.invoke()
    delta = time.time() - startTime
    print("Inference time:", '%.1f' % (delta * 1000), "ms\n")

    scores = interp.get_tensor(output_details[1]['index'])
    bboxes = interp.get_tensor(output_details[0]['index'])

    # can be converted to numpy for other devices
    # using torch here is only for references.
    ori_scores = torch.from_numpy(scores[0])
    ori_bboxes = torch.from_numpy(bboxes)

    # decode bbox cordinates with priors
    decoded_bboxes = simple_bbox_decode(priors, ori_bboxes, strides)[0]
    scores_list = []
    labels_list = []
    bboxes_list = []
    for cls_id in range(len(texts)):
        cls_scores = ori_scores[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = nms(decoded_bboxes, cls_scores, iou_threshold=0.5)
        cur_bboxes = decoded_bboxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)

    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]
    # only for visualization, add an extra NMS
    keep_idxs = nms(bboxes, scores, iou_threshold=nms_thr)
    num_dets = min(len(keep_idxs), max_dets)
    bboxes = bboxes[keep_idxs].unsqueeze(0)
    scores = scores[keep_idxs].unsqueeze(0)
    labels = labels[keep_idxs].unsqueeze(0)

    scores = scores[0, :num_dets].numpy()
    bboxes = bboxes[0, :num_dets].numpy()
    labels = labels[0, :num_dets].numpy()

    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)

    if vis:
        x1, y1, x2, y2 = bboxes[0]

        cropped_image = ori_image[round(y1):round(y2), round(x1):round(x2)]
        #print(cropped_image)

        # Add person id to labels

        if(num_dets > 0):
            #print(cropped_image)
            extracted_features = extract(cropped_image, osnet_ain_model)
            if len(extracted_features) != 0:
                # Add new person if data is empty
                if not dt_prs:
                    dt_prs[f"id_{id}"] = {
                        "extracted_features": extracted_features,
                        "id": id,
                        "camera_id": cam_id,
                        "class_name": labels[0],
                        "bbox": bboxes[0],
                        "confidence": scores[0],                        "identification_color": np.random.randint(0, 255, size=3),
                        }
                    id += 1
                else:
                    # Search Top 1 identification_score person identification
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

                    # Add or replace data for new person profile
                    if best_match["identification_score"] < ftr_thr:
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
                        print("Match")
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
                        print("New")
                        image_out = visualize(ori_image, bboxes, labels, scores, texts, id)
                        id += 1
        #print(extract(image_out))
        cv2.imwrite(osp.join(output_dir, "image.jpg"), image_out)
        #cv2.imshow("Processed Video", image_out)
        print(f"detecting {num_dets} objects.")
        print(dt_prs)
        #return image_out, ori_scores, ori_bboxes[0]
        return dt_prs, id
    else:
        #return bboxes, labels, scores
        return dt_prs, id
    
    '''
    if vis:
        image_out = visualize(ori_image, bboxes, labels, scores, texts)
        cv2.imwrite(osp.join(output_dir, "image.jpg"), image_out)
        #cv2.imshow("Processed Video", image_out)
        print(f"detecting {num_dets} objects.")
        return image_out, ori_scores, ori_bboxes[0]
    else:
        return bboxes, labels, scores
'''


def main():

    args = parse_args()
    ext_delegate = None
    ext_delegate_options = {}

    # parse extenal delegate options
    if args.ext_delegate_options is not None:
        options = args.ext_delegate_options.split(';')
        for o in options:
            kv = o.split(':')
            if (len(kv) == 2):
                ext_delegate_options[kv[0].strip()] = kv[1].strip()
            else:
                raise RuntimeError('Error parsing delegate option: ' + o)

    # load external delegate
    if args.ext_delegate is not None:
        print('Loading external delegate from {} with args: {}'.format(
            args.ext_delegate, ext_delegate_options))
        ext_delegate = [
            tflite.load_delegate(args.ext_delegate, ext_delegate_options)
        ]
    

    yolo_world_tflite_file = args.yolo_world
    osnet_ain_tflite_file = args.osnet_ain
    # init ONNX session
    interpreter = tflite.Interpreter(model_path=yolo_world_tflite_file,
                                      experimental_preserve_all_tensors=True,
                                      experimental_delegates=ext_delegate,
                                      num_threads=args.num_threads)
    interpreter.allocate_tensors()
    print("Init TFLite Interpter")
    output_dir = "onnx_outputs"
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    cam = {}
    videos = np.array(["https://youtu.be/Tn70NxIMk2Q"])

    # Determines how to capture the stream (Is different using youtube videos)
    youtubeVideos = True

    total_cam = len(videos)
    for i in range(total_cam):
        if youtubeVideos == True:
            stream_url = get_youtube_stream_url(videos[i])
            cam[f"cam_{i}"] = cv2.VideoCapture(stream_url)
        else:
            cam[f"cam_{i}"] = cv2.VideoCapture(videos[i])

    '''
    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]
    '''

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines]
    elif args.text.endswith('.json'):
        texts = json.load(open(args.text))
    else:
        texts = [[t.strip()] for t in args.text.split(',')]

    size = (640, 640)
    strides = [8, 16, 32]

    # prepare anchors, since TFLite models does not contain anchors, due to INT8 quantization.
    featmap_sizes = [(size[0] // s, size[1] // s) for s in strides]
    flatten_priors = generate_anchors(featmap_sizes, strides=strides)
    mlvl_strides = [
        flatten_priors.new_full((featmap_size[0] * featmap_size[1] * 1, ),
                                stride)
        for featmap_size, stride in zip(featmap_sizes, strides)
    ]
    flatten_strides = torch.cat(mlvl_strides)

    # Variable for save detected person
    detected_persons = {}
    id = 0

    while(True):
        images = {}
        # Get camera frame
        for i in range(total_cam):
            ret, images[f"image_{i}"] = cam[f"cam_{i}"].read()
            if not ret:
                print(f"Error reading frame from camera {i}")
                continue

        for i in range(total_cam):
            print("Start to inference.")
            #for img in tqdm.tqdm(images[f"image_{i}"]):
            detected_persons, id = inference_per_sample(interpreter,
                                                    osnet_ain_tflite_file,
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
            print("Finish inference")


if __name__ == "__main__":
    main()
