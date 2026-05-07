from io import BytesIO
import os
import boto3
import cv2
import numpy as np
import pydicom

# todo: global var
s3_client = boto3.client(
    "s3",
    endpoint_url="https://storage.intra.ispras.ru",
)


def parse_s3_path(s3_path: str):
    """
    Parse an S3 path to extract the bucket name and image key.

    Parameters
    ----------
    s3_path: str
        The full S3 path in the format 's3://bucket_name/folder1/.../image_key'.

    Returns
    -------
    tuple:
        A tuple containing the bucket name and the full image key (including folders).
    """
    assert s3_path.startswith("s3://"), "S3 path must start with 's3://'"
    s3_parts = s3_path.replace("s3://", "").split("/", 1)
    bucket_name = s3_parts[0]
    image_key = s3_parts[1] if len(s3_parts) > 1 else ""
    return bucket_name, image_key


def load_file_bytes(file_path: str):
    """
    Load image bytes from either local file system or S3.

    Parameters
    ----------
    image_path: str
        The path to the image file, either local or S3.

    Returns
    -------
    BytesIO:
        The image data as a BytesIO object.
    """
    global s3_client

    if file_path.startswith("s3://"):
        # Parse the S3 path
        bucket_name, image_key = parse_s3_path(file_path)
        # Load from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        return BytesIO(response["Body"].read())


    # Load locally
    with open(file_path, "rb") as f:
        return BytesIO(f.read())


def load_pydicom(dicom_path: str):
    """
    Load a DICOM image from either the local file system or an S3 bucket.

    Parameters
    ----------
    dicom_path: str
        The path or key to the DICOM file. Can be a local path or an S3 key.

    Returns
    -------
    pydicom.FileDataset:
        The loaded DICOM object, which can be used to access pixel data or metadata.
    """

    if dicom_path.startswith("s3://"):
        image_bytes = load_file_bytes(dicom_path)
        dicom_ds = pydicom.dcmread(image_bytes, force=True)  # Read DICOM from bytes
    else:
        dicom_ds = pydicom.dcmread(
            dicom_path, force=True
        )  # Read DICOM directly from file path

    return dicom_ds

def load_png(image_path: str) -> np.array:
    """
    Load an image as a NumPy array from local file system or S3.

    Parameters
    ----------
    image_path: str
        The path to the image file, either local or S3.

    Returns
    -------
    np.array:
        The loaded image as a NumPy array in RGB format.
    """
    if image_path.startswith("s3://"):
        # Load from S3
        image_bytes = load_file_bytes(image_path)
        image_bytes.seek(0)  # Ensure the BytesIO object is at the start
        image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        # Load from local file system
        image = cv2.imread(image_path)

    # Convert to RGB format
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        raise ValueError(f"Failed to load image from path: {image_path}")

    return image



def load_image(path):
    """
    Reads and preprocesses a mammography image from a supported file format.

    This function supports both DICOM images (with extensions ".dcm", ".dicom", or no extension)
    and common image formats (".png", ".jpg", ".pgm"). The choice of preprocessing algorithm is
    based on the file extension: if the extension is missing or indicates DICOM, the DICOM-specific
    pipeline is applied.

    All images, regardless of format, are min-max normalized to the [0, 1] range.

    Optionally, the function supports cropping the image to the region of interest (ROI)
    using a provided mask (`path_to_mask`) and a probability threshold (`p_mask`).

    Further custom preprocessing steps can be applied through `additional_mammo_preprocess_kwargs`.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".dcm", ".dicom"] or ext == "":
        dcm = load_pydicom(path)
        pixel_array = dcm.pixel_array
        photometric_interpretation = getattr(
        dcm, "PhotometricInterpretation", "MONOCHROME2")
        if photometric_interpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array
        pixel_array = np.repeat(pixel_array[:, :, np.newaxis], 3, axis=2)
        pixel_array = pixel_array.astype(np.float32)

    elif ext in [".png", ".jpg", ".pgm"]:
        pixel_array = load_png(path)
    else:
        raise ValueError(f"The extension of file {path} is not supported")

    pixel_array -= np.min(pixel_array)
    pixel_array /= np.max(pixel_array)

    return pixel_array
