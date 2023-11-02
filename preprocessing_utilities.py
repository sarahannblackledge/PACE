# Copyright (c) 2023 Moletest (Scotland) Limited t/a nomela.

"""Utilities used for preprocessing of input RGBA images from Nomela system."""

import os

import numpy as np
from numpy.typing import NDArray
import scipy
from tensorflow import keras
from tqdm import tqdm


def check_normalization_input_shape(x: NDArray) -> bool:
    """  Check that input array has correct shape prior to use of normalization functions.

    Checks that input has shape (n_images, rows, columns, 3).

    Args:
         x (NDArray): The array to check

    Returns:
        bool
    """

    if not x.ndim == 4:  # 4 dimensions
        return False

    if not x.shape[-1] == 3:  # 3 color channels
        return False

    return True


def normalization_none(x: NDArray) -> NDArray:
    """ Null image normalization prior to input into relevant pretrained keras model.

    This is a no-op that just returns the input array

    Args:
        x (NDArray): Input image(s) with shape (N, rows, columns, 3)

    Raises:
        ValueError: Occurs when shape of `x` shape is not valid.
    """
    if not check_normalization_input_shape(x):
        raise ValueError(f"x has improper dimensions {x.shape}")

    return x


def normalization_tensorflow(x: NDArray) -> NDArray:
    """ Tensorflow image normalization prior to input into relevant pretrained keras model.

    Performs following operations on R, G and B color channels:
        - R' = R / 127.5 - 1
        - G' = G / 127.5 - 1
        - B' = B / 127.5 - 1

    Args:
        x (NDArray): Input image(s) with shape (N, rows, columns, 3)

    Returns:
        NDArray: The normalized values

    Raises:
        ValueError: Occurs when shape of `x` shape is not valid.
    """
    if not check_normalization_input_shape(x):
        raise ValueError(f"x has improper dimensions {x.shape}")

    return x / 127.5 - 1


def normalization_torch(x: NDArray) -> NDArray:
    """ Torch image normalization prior to input into relevant pretrained keras model.

    Performs following operations on R, G and B color channels:
        - R' = (R / 255 - 0.485) / 0.229
        - G' = (G / 255 - 0.456) / 0.224
        - B' = (B / 255 - 0.406) / 0.225

    Args:
        x (NDArray): Input image(s) with shape (N, rows, columns, 3)

    Returns:
        NDArray: The normalized values

    Raises:
        ValueError: Occurs when shape of `x` shape is not valid.
    """
    if not check_normalization_input_shape(x):
        raise ValueError(f"x has improper dimensions {x.shape}")

    x = x.astype("float")
    x[..., 0] = (x[..., 0] / 255 - 0.485) / 0.229
    x[..., 1] = (x[..., 1] / 255 - 0.456) / 0.224
    x[..., 2] = (x[..., 2] / 255 - 0.406) / 0.225

    return x


def normalization_caffe(x: NDArray) -> NDArray:
    """ Caffe image normalization prior to input into relevant pretrained keras model.

    Performs following operations on R, G and B color channels:
        R' = B - 103.939
        G' = G - 116.779
        B' = R - 123.68

    Args:
        x (NDArray): Input image(s) with shape (N, rows, columns, 3)

    Returns:
        NDArray: The normalized values

    Raises:
        ValueError: Occurs when shape of `x` shape is not valid.
    """
    if not check_normalization_input_shape(x):
        raise ValueError(f"x has improper dimensions {x.shape}")

    x = x.astype("float")
    x[..., 0] = x[..., 0] - 123.68
    x[..., 1] = x[..., 1] - 116.779
    x[..., 2] = x[..., 2] - 103.939
    x = x[..., ::-1]

    return x


def keras_normalization(x: NDArray, kind: str) -> NDArray:
    """ A combined call function for the available keras normalization routines.

    Provides access to the routines via a string rather than calling each independently.

    Args:
        x (NDArray): Input image(s) with shape (N, rows, columns, 3).
        kind (str): The kind of normalization to perform.  Must be one of:
            "tensorflow", "torch", "caffe", or "none" (capitals are ignored).

    Returns:
        NDArray: The normalized values

    Raises:
        ValueError: Occurs when shape of `x` shape is not valid.
        ValueError: Occurs `kind` has unrecognised value.
    """
    if not check_normalization_input_shape(x):
        raise ValueError(f"x has improper dimensions {x.shape}")

    if not kind.lower() in ["tensorflow", "torch", "caffe", "none"]:
        raise ValueError(f"kind has improper value {kind} and must be one of "
                         "'tensorflow', 'torch', 'caffe', or 'none'")

    if kind.lower() == "tensorflow":
        return normalization_tensorflow(x)
    elif kind.lower() == "torch":
        return normalization_torch(x)
    elif kind.lower() == "caffe":
        return normalization_caffe(x)
    else:
        return normalization_none(x)


def get_keras_model_names() -> NDArray:
    """ Return the model names that are current available

    Returns:
        numpy.ndarray: A sorted array of strings.
    """
    return np.sort([model_name for model_name in get_keras_models().keys()])


def get_keras_models() -> dict:
    """ Class constructors for pretrained keras model architectures.

    Returns:
        dict: Keys are the name of each architecture, and each value is a 2-element tuple
            consisting of (keras.models.Model: model class
                           constructor, str: normalization kind)
    """
    return {'ConvNeXtTiny': (keras.applications.ConvNeXtTiny, "None"),
            'ConvNeXtSmall': (keras.applications.ConvNeXtSmall, "None"),
            'ConvNeXtBase': (keras.applications.ConvNeXtBase, "None"),
            'ConvNeXtLarge': (keras.applications.ConvNeXtLarge, "None"),
            'ConvNeXtXLarge': (keras.applications.ConvNeXtXLarge, "None"),
            'DenseNet121': (keras.applications.DenseNet121, "torch"),
            'DenseNet169': (keras.applications.DenseNet169, "torch"),
            'DenseNet201': (keras.applications.DenseNet201, "torch"),
            'EfficientNetB0': (keras.applications.EfficientNetB0, "none"),
            'EfficientNetB1': (keras.applications.EfficientNetB1, "none"),
            'EfficientNetB2': (keras.applications.EfficientNetB2, "none"),
            'EfficientNetB3': (keras.applications.EfficientNetB3, "none"),
            'EfficientNetB4': (keras.applications.EfficientNetB4, "none"),
            'EfficientNetB5': (keras.applications.EfficientNetB5, "none"),
            'EfficientNetB6': (keras.applications.EfficientNetB6, "none"),
            'EfficientNetB7': (keras.applications.EfficientNetB7, "none"),
            'EfficientNetV2B0': (keras.applications.EfficientNetV2B0, "none"),
            'EfficientNetV2B1': (keras.applications.EfficientNetV2B1, "none"),
            'EfficientNetV2B2': (keras.applications.EfficientNetV2B2, "none"),
            'EfficientNetV2B3': (keras.applications.EfficientNetV2B3, "none"),
            'EfficientNetV2S': (keras.applications.EfficientNetV2S, "none"),
            'EfficientNetV2M': (keras.applications.EfficientNetV2M, "none"),
            'EfficientNetV2L': (keras.applications.EfficientNetV2L, "none"),
            'InceptionResNetV2': (keras.applications.InceptionResNetV2, "tensorflow"),
            'InceptionV3': (keras.applications.InceptionV3, "tensorflow"),
            'MobileNet': (keras.applications.MobileNet, "tensorflow"),
            'MobileNetV2': (keras.applications.MobileNetV2, "tensorflow"),
            'MobileNetV3Small': (keras.applications.MobileNetV3Small, "none"),
            'MobileNetV3Large': (keras.applications.MobileNetV3Large, "none"),
            'NASNetMobile': (keras.applications.NASNetMobile, "tensorflow"),
            'NASNetLarge': (keras.applications.NASNetLarge, "tensorflow"),
            'RegNetX002': (keras.applications.RegNetX002, "none"),
            'RegNetX004': (keras.applications.RegNetX004, "none"),
            'RegNetX006': (keras.applications.RegNetX006, "none"),
            'RegNetX008': (keras.applications.RegNetX008, "none"),
            'RegNetX016': (keras.applications.RegNetX016, "none"),
            'RegNetX032': (keras.applications.RegNetX032, "none"),
            'RegNetX040': (keras.applications.RegNetX040, "none"),
            'RegNetX064': (keras.applications.RegNetX064, "none"),
            'RegNetX080': (keras.applications.RegNetX080, "none"),
            'RegNetX120': (keras.applications.RegNetX120, "none"),
            'RegNetX160': (keras.applications.RegNetX160, "none"),
            'RegNetX320': (keras.applications.RegNetX320, "none"),
            'RegNetY002': (keras.applications.RegNetY002, "none"),
            'RegNetY004': (keras.applications.RegNetY004, "none"),
            'RegNetY006': (keras.applications.RegNetY006, "none"),
            'RegNetY008': (keras.applications.RegNetY008, "none"),
            'RegNetY016': (keras.applications.RegNetY016, "none"),
            'RegNetY032': (keras.applications.RegNetY032, "none"),
            'RegNetY040': (keras.applications.RegNetY040, "none"),
            'RegNetY064': (keras.applications.RegNetY064, "none"),
            'RegNetY080': (keras.applications.RegNetY080, "none"),
            'RegNetY120': (keras.applications.RegNetY120, "none"),
            'RegNetY160': (keras.applications.RegNetY160, "none"),
            'RegNetY320': (keras.applications.RegNetY320, "none"),
            'ResNet50': (keras.applications.ResNet50, "caffe"),
            'ResNet101': (keras.applications.ResNet101, "caffe"),
            'ResNet152': (keras.applications.ResNet152, "caffe"),
            'ResNetRS50': (keras.applications.ResNetRS50, "none"),
            'ResNetRS101': (keras.applications.ResNetRS101, "none"),
            'ResNetRS152': (keras.applications.ResNetRS152, "none"),
            'ResNetRS200': (keras.applications.ResNetRS200, "none"),
            'ResNetRS270': (keras.applications.ResNetRS270, "none"),
            'ResNetRS350': (keras.applications.ResNetRS350, "none"),
            'ResNetRS420': (keras.applications.ResNetRS420, "none"),
            'ResNet50V2': (keras.applications.ResNet50V2, "tensorflow"),
            'ResNet101V2': (keras.applications.ResNet101V2, "tensorflow"),
            'ResNet152V2': (keras.applications.ResNet152V2, "tensorflow"),
            'VGG16': (keras.applications.VGG16, "caffe"),
            'VGG19': (keras.applications.VGG19, "caffe"),
            'Xception': (keras.applications.Xception, "tensorflow")}


def features_for_array(x_array: NDArray, batch_size: int = 30, pbar_desc: str = "") \
        -> dict:
    """ Generate features for a given input array

    Args:
        x_array (NDArray): The data from which to generate features.  Should have shape
            (N, 512, 512, 3) and integer values in range 0 to 255.  The latter condition will
            not be checked.
        batch_size (int): The batch size over which to produce the features. This will avoid
            memory issues that are encountered by trying to process large image numbers.
            Defaults to 30.
        pbar_desc (str): The description to use for the progress bar.  If empty, then
            no progress bar is produced. Defaults to an empty string.

    Returns:
        dict: A dictionary of Numpy arrays with shape (N, n_features).  Keys are the names of
            the pretrained keras models from which the features were produced.

    Raises:
        ValueError: Occurs when shape of `x_array` is not (N, 512, 512, 3).
    """

    if not np.allclose(x_array.shape[1::], (512, 512, 3)):
        raise ValueError("Invalid input array shape")

    # The number of input RGB arrays
    n_arrays = x_array.shape[0]

    # The model generators
    keras_models = get_keras_models()

    # Find the total number of iterations
    batch_steps = np.arange(0, n_arrays, batch_size)
    n_batches = len(batch_steps)
    n_models = len(keras_models)
    n_iterations = n_batches * n_models

    # The progress bar
    if pbar_desc == "":
        pbar = None
    else:
        pbar = tqdm(total=n_iterations, desc=pbar_desc)

    # Storage
    features = {}

    # Loop over models
    for model_name in keras_models.keys():
        features_ = []
        model = keras_models[model_name][0](include_top=False,
                                            weights="imagenet",
                                            pooling="max",
                                            input_shape=(512, 512, 3))
        model.trainable = False
        normalization_kind = keras_models[model_name][1]

        # Loop over batches
        for batch in range(n_batches - 1):
            x_array_sub = x_array[batch_steps[batch]:batch_steps[batch + 1], ...]
            x_array_sub_processed = keras_normalization(x_array_sub, normalization_kind)
            features_.append(model(x_array_sub_processed))
            if pbar:
                pbar.update(1)

        # The final batch
        x_array_sub = x_array[batch_steps[-1]::, ...]
        x_array_sub_processed = keras_normalization(x_array_sub, normalization_kind)
        features_.append(model(x_array_sub_processed))
        if pbar:
            pbar.update(1)

        # Add to feature results
        features[model_name] = np.vstack(features_)

    return features


def calc_padding(input_shape: tuple, output_shape: tuple):
    """ Calculate the padding array required to create output_shape from input_shape

    Example:
        To pad an array with shape (20, 20) to (30, 33) would require pads of
        (5, 5) in the first axis and (7, 6) in the second axis. Negative padding is allowed.

    Args:
        input_shape (tuple): A n-element tuple defining the shape of the input array.
        output_shape (tuple): A n-element tuple defining the shape of the output array.

    Returns:
        list: An n-element list of 2-element tuples defining the padding for each dimension.
    """
    padding = []
    for input_dimension, output_dimension in zip(input_shape, output_shape):
        diff = output_dimension - input_dimension
        if diff % 2 == 1:
            padding.append((int(diff / 2) + np.sign(diff), int(diff / 2)))
        else:
            padding.append((int(diff / 2), int(diff / 2)))
    return padding


def pad_rgb_array(input_array: NDArray, output_shape: tuple, constant_value: float = 0.0):
    """ Pad a rgba array to desired output shape

    If `output_shape` is smaller than the input shape in any axis, that axis will be symmetrically
        cropped.

    Args:
        input_array (NDArray): The array to pad, with shape (n_1, n_2, k) where K represents
            the color channel.
        output_shape (tuple): The desired output shape (m_1, m_2).
        constant_value (float): The constant value to use in the padded array. Default = 0.0.

    Returns:
        numpy.ndarray: The padded output.
    """
    n_1, n_2, _ = input_array.shape
    padding = calc_padding((n_1, n_2), output_shape)
    padding_0 = padding[0]
    padding_1 = padding[1]

    # We need to deal with negative pads, which numpy doesn't
    padding_numpy_0 = padding_0
    padding_numpy_1 = padding_1
    if padding_0[0] < 0:
        padding_numpy_0 = (0, 0)
    if padding_1[0] < 0:
        padding_numpy_1 = (0, 0)

    # Generate the padded image
    output_array = np.pad(input_array,
                          (padding_numpy_0, padding_numpy_1, (0, 0)),
                          mode="constant",
                          constant_values=constant_value)

    # Deal with negatives
    if padding_0[0] < 0:
        output_array = output_array[-padding_0[0]:padding_0[1], :, :]
    if padding_1[0] < 0:
        output_array = output_array[:, -padding_1[0]:padding_1[1], :]
    return output_array


def rotate_rgb_array(input_array: NDArray, radians: float = None):
    """ Rotate a rgb array

    Args:
        input_array (NDArray): The array to rotate.  This assumes that the color channel is
            the last index.
        radians (float): The desired rotation. If None (default), then one is randomly chosen
            (uniform in range 0 -> 2*pi).

    Returns:
        numpy.ndarray: The rotated output array.
    """
    if radians is None:
        radians = np.random.rand() * 2 * np.pi
    degrees = radians * 180. / np.pi
    output_array = scipy.ndimage.rotate(input_array,
                                        degrees,
                                        order=0,  # Linear interpolation
                                        axes=(1, 0),
                                        reshape=False,
                                        mode="nearest")
    return output_array


def downsample_rgb_array(input_array: NDArray, ds_factor: int):
    """ Downsample a rgba array

    Args:
        input_array (NDArray): The array to downsample. This assumes that the color channel
            is the last index
        ds_factor (int): The downsample factor.

    Returns:
        numpy.ndarray: The downsampled output
    """
    output_array = input_array[::ds_factor, ::ds_factor, :]
    return output_array


def random_flip_rgb_array(input_array: NDArray, prob: float = 0.5, axis: int = None):
    """ Randomly flip a rgb array along a given axis in the image dimensions.

    Args:
        input_array (NDArray): The array to flip.  This assumes that the color channel is the
            last index
        prob (float): The probability with which to flip (default = 0.5).
        axis (int): Valid values are 0, 1 or None.  If None (default), then one will be chosen at
            random (probability provided by `prob')

    Returns:
        numpy.ndarray: The flipped output.
    """
    if axis is None:
        if np.random.rand() > prob:
            axis = 0
        else:
            axis = 1
    output_array = input_array.copy()
    if np.random.rand() > prob:
        if axis == 0:
            output_array = input_array[::-1, :, :]
        else:
            output_array = input_array[:, ::-1, :]
    return output_array


def preprocess_rgba_array(rgba_array: NDArray, augment: bool = False,
                          normalization_kind: str = "none") -> NDArray:
    """ A utility function to extract an input RGB image file.

    This function will:
        1. Remove the alpha channel (if present - last dimensions has 4 channels)
        2. Pad the image to 512*3 by 512*3 (to allow rotations)
        3. Perform random flipping and rotation (if `augment`=True)
        4. Downsample the image back to 512 by 512.
            (Provided input images have the same resolution, this ensures
            that output resolutions are also the same)

    Args:
        rgba_array (NDArray): An RGBA array loaded from the file, with shape (n_1, n_2, 4).
        augment (bool): Whether to randomly rotate / flip the image (default = False)
        normalization_kind (str): The kind of normalization to perform.  Must be one of:
            "tensorflow", "torch", "caffe", or "none" (capitals are ignored).
            Use "none" of no normalization should be performed.

    Returns:
        NDArray: The processed image with shape (512, 512, 3)
    """
    output_array = rgba_array.copy().astype("float")

    # Check dimensions
    if not output_array.ndim == 3:
        raise ValueError(f"Input array has invalid dimensions = {output_array.ndim}")

    # Ensure the last channel is valid shape
    _, _, channels = output_array.shape
    if not channels == 4:
        raise ValueError(f"Input image file incorrect number of channels = {channels}")

    # Check normalization kind is valid
    if not normalization_kind.lower() in ["tensorflow", "torch", "caffe", "none"]:
        raise ValueError(f"kind has improper value {normalization_kind} and must be one of "
                         "'tensorflow', 'torch', 'caffe', or 'none'")

    # Remove the alpha channel
    output_array = output_array[:, :, 0:-1]

    # Pad to 512 * 4
    output_array = pad_rgb_array(output_array, (512 * 4, 512 * 4))

    # Downsample to 512 x 512
    output_array = downsample_rgb_array(output_array, 4)

    # Augment if requested
    if augment:
        # Random rotate
        output_array = rotate_rgb_array(output_array)
        # Flip
        output_array = random_flip_rgb_array(output_array)

    # Keras normalization - Note that we need to temporarily make it 4D.
    output_array = keras_normalization(output_array[None, ...], normalization_kind)
    output_array = output_array[0]

    return output_array


def image_file_for_patient_id(patient_id: str, patient_diagnosis: str,
                              raw_data_directory: str) -> str:
    """ Determine the raw image file location for a patient ID

    Args:
        patient_id (str): The patient ID
        patient_diagnosis (str): The patient diagnosis
        raw_data_directory (str): Where the raw data are located

    Returns:
        str: The file location

    Raises:
        FileNotFoundError: Occurs when no file is found.
    """
    patient_image_directory = os.path.join(raw_data_directory,
                                           patient_diagnosis,
                                           patient_id)

    if not os.path.exists(patient_image_directory):
        raise FileNotFoundError(f"Could not find data directory for patient {patient_id}")

    fname = os.listdir(patient_image_directory)[0]
    if "mask-filter.png" not in fname:
        raise FileNotFoundError(f"Incorrect image filename found for patient {patient_id}")

    return os.path.join(patient_image_directory, fname)
