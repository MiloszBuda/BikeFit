import numpy as np
import pytest
from bikefit.models.rtmpose import RTMPoseEstimator


@pytest.mark.integration
def test_rtmpose_real_inference_blank_image():
    """Verify RTMPose loads ONNX model and runs inference on a real numpy array."""
    estimator = RTMPoseEstimator(mode="lightweight", device="cpu")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    kpts, scores = estimator.predict(dummy_img)

    assert isinstance(kpts, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert kpts.shape[-1] == 2
    assert scores.shape[-1] == 26
