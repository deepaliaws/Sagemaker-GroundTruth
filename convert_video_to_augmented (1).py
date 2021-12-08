import json
from io import BytesIO
from typing import List, Tuple, Union
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from PIL import Image

OBJECT_OUTPUT_REF = "object-output-ref"
retry_config = Config(retries={"max_attempts": 10, "mode": "standard"})
s3_client = boto3.client("s3", config=retry_config)


def get_bucket_and_key_from_uri(s3_uri):
    """
    Converts an S3 object URI to an S3 bucket and S3 key.
    """
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


def get_s3_bytes(s3uri: str) -> BytesIO:
    """
    Returns the object's content in Bytes. If the s3uri is invalid, throws NoSuchKeyError
    :param s3uri: an object s3 uri
    :return: Bytes of content stored in the object s3 uri
    """
    bucket, key = get_bucket_and_key_from_uri(s3uri)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return BytesIO(obj["Body"].read())


def get_img_from_s3(s3uri: str) -> Image:
    """
    Returns the object's content in PIL Image format. If the s3uri is invalid, throws NoSuchKeyError
    :param s3uri: an object s3 uri
    :return: returns the content stored in the object s3 uri in PIL image format
    """
    img = Image.open(get_s3_bytes(s3uri))
    return img


def get_image_size(image_uri: str) -> Tuple[int, int, int]:
    """Find dimensions of image file in s3.

    :param image_uri: location of image in s3, e.g. s3://bucket/location/image.jpg
    :return: tuple with (number of channels, height, width)
    """
    img = get_img_from_s3(image_uri)
    w, h = img.size
    c = len(img.getbands())
    return c, h, w


def get_json_from_s3(s3uri: str) -> Union[dict, List]:
    """
    Returns the object's content in JSON format. If the s3uri is invalid, throws NoSuchKeyError
    :param s3uri: an object s3 uri
    :return: JSON formatted output of the content stored in the object s3 uri
    """
    return json.load(get_s3_bytes(s3uri))


def _make_entry(
    source_prefix: str,
    input_frame: dict,
    output_annotations: List[dict],
    original_metadata: dict,
    label_attribute_name: str,
    image_size: Tuple[int, int, int],
) -> dict:
    """Make an entry for an object detection augmented manifest.

    :param source_prefix: s3 prefix (including bucket) of frames in object tracking job
    :param input_frame: data from source video json for a single frame
    :param output_annotations: data with bounding box info of all annotated frames within the video
    :param original_metadata: metadata from the video tracking job
    :param label_attribute_name: name of label field in video output manifest, e.g. Person-ref
    :param image_size: tuple with (number of channels, height, width)
    :return: dictionary corresponding to line of output manifest
    """
    source_ref = source_prefix + input_frame["frame"]
    annotations = []
    output_frame_tracking_annotations = next(
        (t_a for t_a in output_annotations if t_a["frame"] == input_frame["frame"]),
        None,
    )

    # output_frame_tracking_annotations will be None for frames in the video which have no annotations from QC step
    if output_frame_tracking_annotations is not None:
        for old_annotation in output_frame_tracking_annotations["annotations"]:
            annotation = old_annotation.copy()
            annotation["class_id"] = annotation.pop("class-id")
            annotations.append(annotation)

    c, h, w = image_size
    metadata = original_metadata.copy()
    metadata["label-attribute-name"] = label_attribute_name
    return {
        "source-ref": source_ref,
        "bounding-box": {
            "annotations": annotations,
            "image_size": [{"depth": int(c), "height": int(h), "width": int(w)}],
        },
        "bounding-box-metadata": metadata,
    }


def read_augmented_manifest(s3uri: str) -> List[dict]:
    """
    Returns the GT formatted manifest content from s3uri, if the s3uri is invalid - throws NoSuchKey exception
    :param s3uri: an object s3 uri which contains the manifest
    :return: List of single line json manifest objects
    """
    buf = get_s3_bytes(s3uri)
    return [json.loads(line) for line in buf]


def _make_entry_helper(datum: tuple) -> dict:
    """Helper for multiprocessing creation of output manifest.

    :param datum: Tuple of the inputs for `_make_entry`
    :return: dictionary corresponding to line of output manifest
    """
    return _make_entry(*datum)


def _get_first_frame_size(source_prefix, input_frames):
    """Get the size of the first frame in the video.

    :param source_prefix: s3 prefix (including bucket) of frames in object tracking job
    :param input_frames: list of frames from source video json data
    :return: tuple with (number of channels, height, width)
    """
    source_ref = source_prefix + input_frames[0]["frame"]
    return get_image_size(source_ref)


def _make_od_manifest(
    source_seq: dict,
    output_seq: dict,
    original_metadata: dict,
    label_attribute_name: str,
) -> List[dict]:
    """Make input manifest for object detection training (helper function).

    :param source_seq: dictionary from source video source json
    :param output_seq: dictionary from source video json containing the frame data
    :param original_metadata: dictionary from metadata in video output manifest
    :param output_manifest_s3_uri: s3 uri of video output manifest
    :param label_attribute_name: name of label field in video output manifest, e.g. Person-ref
    :return: list corresponding to the lines of the output manifest
    """
    source_prefix = source_seq["prefix"]
    input_frames = source_seq["frames"]
    output_annotations = output_seq.get("tracking-annotations", [])

    image_size = _get_first_frame_size(source_prefix, input_frames)
    data = [
        (
            source_prefix,
            input_frame,
            output_annotations,
            original_metadata,
            label_attribute_name,
            image_size,
        )
        for input_frame in input_frames
    ]
    manifest = [_make_entry_helper(datum) for datum in data]

    return manifest


def make_od_input_manifest(
    label_attribute_name: str, output_manifest_s3_uri: str, class_map: dict = None
) -> List[dict]:
    """Make input manifest for object detection training with augmented output manifest of GT video object tracking job.

    :param label_attribute_name: name of the video object tracking labeling job
    :param output_manifest_s3_uri: augmented output manifest of the labeling job
    :param class_map: exhaustive class map consisting of all categories (not just the ones annotated by humans)
    :return: list corresponding to the lines of the output manifest
    """
    output_manifests = read_augmented_manifest(output_manifest_s3_uri)
    augmented_mv_input_manifest = []
    for video_output_manifest in output_manifests:
        video_source_ref = video_output_manifest["source-ref"]
        output_seq_uri = video_output_manifest[label_attribute_name]
        original_metadata = video_output_manifest[label_attribute_name + "-metadata"]

        # Use the exhaustive class map having all class-labels for Machine validation. GT strips class-labels which
        # aren't used for annotation in the output of labeling jobs. So, if humans missed a
        # class using class-map from GT output would make MV skip it.
        if class_map:
            original_metadata["class-map"] = class_map

        source_seq = get_json_from_s3(video_source_ref)
        output_seq = get_json_from_s3(output_seq_uri)
        augmented_mv_input_manifest.extend(
            _make_od_manifest(
                source_seq,
                output_seq,
                original_metadata,
                label_attribute_name,
            )
        )

    return augmented_mv_input_manifest
