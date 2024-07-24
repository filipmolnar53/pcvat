# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import zipfile

from datumaro.components.dataset import Dataset
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.plugins.coco_format.importer import CocoImporter
from datumaro.components.annotation import Points

from cvat.apps.dataset_manager.bindings import GetCVATDataExtractor, detect_dataset, \
    import_dm_annotations
from cvat.apps.dataset_manager.util import make_zip_archive

from .registry import dm_env, exporter, importer

# my own logger config for debugging
import logging
import json
import os
# import debugpy

my_export_logger = logging.getLogger("my_export_logger")
my_export_logger.setLevel(logging.DEBUG)

my_fh = logging.FileHandler('/home/django/logs/my_log.log', encoding='utf-8')
my_fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
my_fh.setFormatter(formatter)
my_export_logger.addHandler(my_fh)

my_export_logger.info('first--')

@importer(name="COCO Photoneo", version="1.0", ext="ZIP")
def my_importer(src_file, temp_dir, instance_data, load_data_callback=None, **kwargs):
    if zipfile.is_zipfile(src_file):
        zipfile.ZipFile(src_file).extractall(temp_dir)
        # We use coco importer because it gives better error message
        detect_dataset(temp_dir, format_name='coco', importer=CocoImporter)
        dataset = Dataset.import_from(temp_dir, 'coco_instances', env=dm_env)
        if load_data_callback is not None:
            load_data_callback(dataset, instance_data)

        json_path = temp_dir+"/annotations/instances_default.json"

        with open(json_path, 'r') as file:
            data = json.load(file)

        label_categories = dataset.categories().get(AnnotationType.label, None)
        for category in data["categories"]:
            for kp in category["keypoints"]:
                label_categories.add(category["name"]+kp)

        dataset.categories()[AnnotationType.label] = label_categories
        my_export_logger.info(f"dataset_categories: {dataset.categories().get(AnnotationType.label, None)}")

        for ann in data["annotations"]:
            file_name = find_property_by_id(data["images"], ann["image_id"], "file_name")
            keypoints = find_property_by_id(data["categories"], ann["category_id"], "keypoints")
            ann_label = find_property_by_id(data["categories"], ann["category_id"], "name")
            my_export_logger.info(f"ann : {ann_label}, {keypoints}")
            my_export_logger.info(f"keypoints: {ann}")
            for i, kp in enumerate(keypoints):
                x = ann["keypoints"][i*3]
                y = ann["keypoints"][i*3+1]
                vis = ann["keypoints"][i*3+2]

                if vis:
                    item = dataset.get(file_name.split(".")[0])
                    label_index = label_categories.find(ann_label+kp)
                    my_export_logger.info(f"label_index: {label_index[0]}")
                    item.annotations.append(Points([x,y], label=label_index[0]))
                    my_export_logger.info(f"item: {item}")
                    dataset.put(item)
                    # my_export_logger.info(f"new item: {dataset.get(file_name.split(".")[0])}")
        import_dm_annotations(dataset, instance_data)
    else:
        dataset = Dataset.import_from(src_file.name,
            'coco_instances', env=dm_env)
        import_dm_annotations(dataset, instance_data)


def get_image_id(images, file_name):
    for image in images:
        if image["file_name"] == file_name:
            return image["id"]
    return None

def bbox_to_flat_polygon(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y, x+w, y+h, x, y+h]

def is_point_in_polygon(point, flat_polygon):
    """
    Check if a point is inside a polygon using the ray-casting algorithm.

    Args:
    point: A tuple (x, y) representing the coordinates of the point.
    polygon: A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing the vertices of the polygon.
    Returns:
    True if the point is inside the polygon, False otherwise.
    """
    polygon = [(flat_polygon[i], flat_polygon[i + 1]) for i in range(0, len(flat_polygon), 2)]

    x, y = point
    num_vertices = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(num_vertices + 1):
        p2x, p2y = polygon[i % num_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    # my_export_logger.info('testing kp: {}, if it is inside flat polygon: {}'.format(str(point), str(flat_polygon)))
    # if inside:
    #     my_export_logger.info('jupiii! Is inside flat polygon')

    return inside

def create_categories(data):
    return data

def find_prefix(string, prefixes):
    for prefix in prefixes:
        if string.startswith(prefix):
            return prefix
    return None


def find_property_by_id(categories, id_to_find, property):
    for category in categories:
        my_export_logger.info(f'===cat: {category}')
        if category['id'] == id_to_find:
            return category[property]
    return None

def remove_prefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

def insert_keypoint_triplet(data, i, idx, kp):
    data["annotations"][i]["num_keypoints"] += 1
    # insert a keypoint to correct triple position in 'keypoints': (x, y, visibility)
    data["annotations"][i]["keypoints"][3*idx] = kp["kp"][0]
    data["annotations"][i]["keypoints"][3*idx + 1] = kp["kp"][1]
    data["annotations"][i]["keypoints"][3*idx +2 ] = 2
    return data

@exporter(name="COCO Photoneo", version="1.0", ext="ZIP")
def my_exporter(dst_file, temp_dir, instance_data, save_images=False):
    with GetCVATDataExtractor(instance_data, include_images=save_images) as extractor:
        dataset = Dataset.from_extractors(extractor, env=dm_env)
        dataset.export(temp_dir, 'coco_instances', save_images=save_images,
            merge_images=True)

    json_path = temp_dir+"/annotations/instances_default.json"

    with open(json_path, 'r') as file:
        data = json.load(file)



    my_export_logger.info('BY FRAMES')

    # collect all categories labels
    # create categories object
    categories = {}
    categories_labels = []
    for shape in instance_data.shapes:
        if shape.type == "rectangle" or shape.type == "polygon":
            if shape.label not in categories:
                # cvat indexes from 1
                categories[shape.label] = {"id": len(categories)+1, "name": shape.label, "keypoints": set()}
                categories_labels.append(shape.label)

    categories_labels = sorted(categories_labels, key=len, reverse=True)
    points_data = {key: [] for key in categories_labels}
    my_export_logger.info('categories: {}'.format(categories))
    my_export_logger.info('categories labels: {}'.format(str(categories_labels)))

    # extract all keypoints, assign frame name and label to them
    for frame_data in instance_data.group_by_frame(include_empty=True):
        for shape in frame_data.labeled_shapes:
            my_export_logger.info('shape: {}'.format(shape))
            if shape.type == "points":
                prefix = find_prefix(shape.label, categories_labels)
                my_export_logger.info('prefix: {}'.format(str(prefix)))

                l = remove_prefix(shape.label, prefix)
                points_data[prefix].append({"kp": shape.points, "label":shape.label, "image_id": get_image_id(data["images"], frame_data.name)})
                categories[prefix]["keypoints"].add(l)


    # format keypoints categories to sorted list
    for label_key in categories.keys():
        categories[label_key]["keypoints"] = list(sorted(categories[label_key]["keypoints"]))

    # reindex annotations to new categories
    # create 'num_keypoints' and 'keypoints' fields
    for i, ann in enumerate(data["annotations"]):
        label = find_property_by_id(data["categories"], ann["category_id"], "name")
        data["annotations"][i]["category_id"] = categories[label]["id"]
        data["annotations"][i]["num_keypoints"] = 0
        data["annotations"][i]["keypoints"] = [0] * (3 * len(categories[label]["keypoints"]))

    data["categories"] = list(categories.values())

    for i, ann in enumerate(data["annotations"]):
        label = find_property_by_id(data["categories"], ann["category_id"], "name")

        for kp in points_data[label]:
            my_export_logger.info('keypoints: {}'.format(kp))
            if ann["image_id"] == kp["image_id"]:
                idx = categories[label]["keypoints"].index(remove_prefix(kp["label"], label))
                if ann["segmentation"] and is_point_in_polygon(kp["kp"], ann["segmentation"][0]):
                    data = insert_keypoint_triplet(data, i, idx, kp)
                elif ann["bbox"] and is_point_in_polygon(kp["kp"], bbox_to_flat_polygon(ann["bbox"])):
                    data = insert_keypoint_triplet(data, i, idx, kp)


    my_export_logger.info('json data: \n{}'.format(json.dumps(data, indent=4)))

    with open(json_path, 'w') as file:
       json.dump(data, file)

    make_zip_archive(temp_dir, dst_file)