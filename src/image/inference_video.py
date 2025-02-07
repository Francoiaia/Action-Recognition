# -*- coding: utf-8 -*-
import torch
import cv2
import time
import argparse
import torchvision.transforms as transforms
import pathlib
import os
import torch.nn.functional as F
import numpy as np

from src.model import build_model
from class_names import class_names as CLASS_NAMES

# construct the argumet parser to parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default="input/video_1.mp4", help="path to the input video")
parser.add_argument(
    "-w",
    "--weights",
    default="outputs/best_model_ovh.pth",
    help="path to the model weights",
)
args = parser.parse_args()

OUT_DIR = "../../outputs/inference_results/video_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# set the computation device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_RESIZE = 224


# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return test_transform


transform = get_test_transform(IMAGE_RESIZE)

weights_path = pathlib.Path(args.weights)
checkpoint = torch.load(weights_path)
# Load the model.
model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

cap = cv2.VideoCapture(0)
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# define the outfile file name
save_name = f"{args.input.split('/')[-1].split('.')[0]}"
# define codec and create VideoWriter object
out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))
# to count the total number of frames iterated through
frame_count = 0
# to keep adding the frames' FPS
total_fps = 0
while cap.isOpened():
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # apply transforms to the input image
        input_tensor = transform(rgb_frame)
        # add the batch dimensionsion
        input_batch = input_tensor.unsqueeze(0)

        # move the input tensor and model to the computation device
        input_batch = input_batch.to(DEVICE)
        model.to(DEVICE)

        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_batch)
            end_time = time.time()

        # get the softmax probabilities
        probabilities = F.softmax(outputs, dim=1).cpu()
        # get the top 1 prediction
        # top1_prob, top1_catid = torch.topk(probabilities, k=1)
        output_class = np.argmax(probabilities)

        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        cv2.putText(frame, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{CLASS_NAMES[int(output_class)]}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Result", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
