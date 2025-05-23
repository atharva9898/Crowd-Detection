#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import torch
import numpy as np
import csv
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# In[2]:


VIDEO_PATH = 'dataset_video.mp4'  
OUTPUT_VIDEO_PATH = 'output_annotated_video.mp4'
OUTPUT_CSV_PATH = 'crowd_log.csv'


# In[ ]:


MODEL_NAME = 'yolov5s'  
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to consider a detection
PERSON_CLASS_INDEX = 0  # COCO class index for 'person'


# In[4]:


# Crowd Definition Parameters
MIN_PEOPLE_FOR_CROWD = 3
MIN_CONSECUTIVE_FRAMES_FOR_CROWD = 10
# Distance threshold for people to be "close". This might need tuning.
# Consider average bbox height or a fixed pixel value. Let's start with fixed.
DISTANCE_THRESHOLD_PIXELS = 75


# In[5]:


def get_bounding_box_center(box):
    """Calculates the center of a bounding box (x_center, y_center)."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# In[ ]:


def find_person_groups(person_detections, distance_threshold):

    if not person_detections:
        return []

    num_persons = len(person_detections)
    if num_persons < 2: # Need at least 2 people to form a potential group connection
        if num_persons == 1: # A single person is a group of 1
            return [[0]]
        return []


    centers = np.array([get_bounding_box_center(p[:4]) for p in person_detections])

    # Calculate pairwise distances between all centers
    dist_matrix = cdist(centers, centers)

    # Create an adjacency matrix: 1 if distance < threshold, 0 otherwise
    adjacency_matrix = dist_matrix < distance_threshold

    # Find connected components (groups)
    # csr_matrix is efficient for sparse matrices
    graph = csr_matrix(adjacency_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    groups = [[] for _ in range(n_components)]
    for person_idx, group_label in enumerate(labels):
        groups[group_label].append(person_idx)

    return groups



# In[ ]:


def main():
    # Load YOLOv5 model
    try:
        model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True)
        print(f"Successfully loaded YOLOv5 model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        print("Please ensure you have an internet connection and torch/torchvision installed.")
        return

    # Setup video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    # Setup CSV writer
    csv_file = open(OUTPUT_CSV_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['FrameNumber', 'NumberOfPeopleInCrowd'])

    frame_number = 0
    consecutive_potential_crowd_frames = 0
    # Stores the size of the largest group qualifying as a crowd in the current sequence
    current_sequence_max_crowd_size = 0

    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number % (fps*2) == 0 : # Log progress every 2 seconds of video
            print(f"Processing frame {frame_number}...")

        # Perform inference
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Detections for the first (and only) image

        # Filter for 'person' class and confidence
        person_detections = []
        for _, det in detections.iterrows():
            if det['name'] == 'person' and det['confidence'] >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                person_detections.append((x1, y1, x2, y2, det['confidence'], int(det['class']), det['name']))

        frame_has_potential_crowd = False
        largest_potential_crowd_size_this_frame = 0

        if person_detections:
            person_groups_indices = find_person_groups(person_detections, DISTANCE_THRESHOLD_PIXELS)

            for group_indices in person_groups_indices:
                if len(group_indices) >= MIN_PEOPLE_FOR_CROWD:
                    frame_has_potential_crowd = True
                    if len(group_indices) > largest_potential_crowd_size_this_frame:
                        largest_potential_crowd_size_this_frame = len(group_indices)
                    # For simplicity, we'll just consider if *any* group meets the criteria
                    # and use the largest one for logging.

        # Update consecutive crowd frame counter
        if frame_has_potential_crowd:
            consecutive_potential_crowd_frames += 1
            # Keep track of the largest crowd size seen during this potential sequence
            current_sequence_max_crowd_size = max(current_sequence_max_crowd_size, largest_potential_crowd_size_this_frame)
        else:
            consecutive_potential_crowd_frames = 0
            current_sequence_max_crowd_size = 0 # Reset when sequence breaks

        # Check for confirmed crowd
        is_crowd_detected_in_frame = (consecutive_potential_crowd_frames >= MIN_CONSECUTIVE_FRAMES_FOR_CROWD)

        # --- Annotation and Logging ---
        annotated_frame = frame.copy()

        # Draw bounding boxes for all detected people
        for x1, y1, x2, y2, conf, _, _ in person_detections:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box



        if is_crowd_detected_in_frame:
            if largest_potential_crowd_size_this_frame >= MIN_PEOPLE_FOR_CROWD:
                 csv_writer.writerow([frame_number, largest_potential_crowd_size_this_frame])

            # Annotate video frame
            cv2.putText(annotated_frame, "CROWD DETECTED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

        # Write the frame to the output video
        out_video.write(annotated_frame)


    # Release resources
    cap.release()
    out_video.release()
    csv_file.close()
    # cv2.destroyAllWindows() # If using imshow

    print(f"Processing complete. Annotated video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"Crowd log saved to: {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    # Create a dummy video for testing if you don't have one
    # This part is optional and for self-testing
    import os
    if not os.path.exists(VIDEO_PATH):
        print(f"Warning: Video file '{VIDEO_PATH}' not found. Creating a dummy video for testing.")
        # Create a dummy 14-second video (e.g., 25 FPS * 14 seconds = 350 frames)
        dummy_fps = 25
        dummy_duration_sec = 14
        dummy_width, dummy_height = 640, 480
        dummy_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        dummy_writer = cv2.VideoWriter(VIDEO_PATH, dummy_fourcc, dummy_fps, (dummy_width, dummy_height))
        for _ in range(dummy_fps * dummy_duration_sec):
            frame = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
            # You could draw some moving shapes to simulate people if you want more complex test
            cv2.putText(frame, "Dummy Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            dummy_writer.write(frame)
        dummy_writer.release()
        print(f"Dummy video '{VIDEO_PATH}' created.")

    main()

