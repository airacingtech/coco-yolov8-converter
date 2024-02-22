# @author Trinity Chung trinityc@berkeley.edu
# Modified ultralytics.data.converter convert_coco

import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import os

import numpy as np
import argparse
import glob
import shutil

def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle['counts']
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    pos = 0
    for index, length in enumerate(s):
        if index % 2 == 1:
            img[pos:pos+length] = 1
        pos += length
    return img.reshape(shape[::-1]).T


def find_contours(binary_mask):
    """
    Find contours in a binary mask.
    binary_mask: 2D numpy array of shape (height, width), with 1s for the object and 0s for the background.
    Returns a list of contours, each contour is a numpy array of shape (n, 1, 2).
    """
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if len(contour) >= 3]

def normalize_points(points, img_dim):
    """
    Normalize points to be relative to image dimensions.
    points: List of (x, y) tuples.
    img_dim: Tuple (image width, image height).
    Returns list of normalized (x, y) tuples.
    """
    normalized_points = []
    for x, y in points:
        normalized_points.extend([x / img_dim[1], y / img_dim[0]])
    return normalized_points


def yolo_format_bbox(bbox, img_dim):
    """
    Convert bounding box to YOLO format.
    bbox: A list of four integers [x_min, y_min, x_max, y_max].
    img_dim: Tuple (image width, image height).
    Returns a list: [x_center, y_center, width, height].
    """
    dw = 1./img_dim[0]
    dh = 1./img_dim[1]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]

def get_bbox_from_mask(mask):
    """
    Get the bounding box from a binary mask.
    mask: 2D numpy array where 1 represents the object and 0 represents the background.
    Returns a list: [x_min, y_min, x_max, y_max].
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max, y_max]


def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def convert_coco(
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
):
    """
    Converts COCO dataset annotations to a YOLO annotation format  suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.

    Example:
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco('../datasets/coco/annotations/', use_segments=True, use_keypoints=False, cls91to80=True)
        ```

    Output:
        Generates output files in the specified output directory.
    """

    # Create dataset directory
    save_dir = increment_path(save_dir)
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)

    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        fn = (
            # Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")
            Path(save_dir) / "labels"
        ) 
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file, "r") as f:
            data = json.load(f)

        images = {f'{x["id"]:d}': x for x in data["images"]}
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        for img_id, anns in tqdm(imgToAnns.items()):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann["iscrowd"]:
                    binary_mask = rle_decode(ann['segmentation'], ann['segmentation']['size'])
                    contours = find_contours(binary_mask)
                    if len(contours) > 0:
                        largest_contour = max(contours, key=cv2.contourArea)
                        yolo_segment = normalize_points(largest_contour.squeeze(), ann['segmentation']['size'])
                        bboxes.append([ann['category_id']] + yolo_segment) 
                else:
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(ann["bbox"], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue

                    cls = ann["category_id"] - 1
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                        if use_segments and ann.get("segmentation") is not None:
                            if len(ann["segmentation"]) == 0:
                                segments.append([])
                                continue
                            elif len(ann["segmentation"]) > 1:
                                s = merge_multi_segment(ann["segmentation"])
                                s = (
                                    (np.concatenate(s, axis=0) / np.array([w, h]))
                                    .reshape(-1)
                                    .tolist()
                                )
                            else:
                                s = [
                                    j for i in ann["segmentation"] for j in i
                                ]  # all segments concatenated
                                s = (
                                    (np.array(s).reshape(-1, 2) / np.array([w, h]))
                                    .reshape(-1)
                                    .tolist()
                                )
                            s = [cls] + s
                            segments.append(s)
                        if use_keypoints and ann.get("keypoints") is not None:
                            keypoints.append(
                                box
                                + (
                                    np.array(ann["keypoints"]).reshape(-1, 3)
                                    / np.array([w, h, 1])
                                )
                                .reshape(-1)
                                .tolist()
                            )

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i, bbox in enumerate(bboxes):
                    if use_keypoints:
                        line = (*(keypoints[i]),)
                    else:
                        line = (
                            *(
                                segments[i]
                                if use_segments and len(segments[i]) > 0
                                else bbox
                            ),
                        )
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

    print(f"COCO data converted successfully.\nResults saved to {save_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('-i', '--handle_images', choices=['copy', 'move', 'none'] , default='copy')
    args = parser.parse_args()

    for d in glob.glob(os.path.join(args.directory, '*')):
        if os.path.isdir(d):
            convert_coco(f"./{d}/annotations", f"./converted/{os.path.basename(d)}")
            if args.handle_images == 'move':
                shutil.move(f"./{d}/images", f"./converted/{os.path.basename(d)}")
            elif args.handle_images == 'copy':
                shutil.copytree(f"./{d}/images", f"./converted/{os.path.basename(d)}/images")

