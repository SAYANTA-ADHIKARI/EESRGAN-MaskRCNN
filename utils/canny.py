import math

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.tensor import Tensor

from kornia.color import rgb_to_grayscale

from kornia import gaussian_blur2d, spatial_gradient
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt

def get_canny_nms_kernel(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    """Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )

def get_hysteresis_kernel(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    """Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )

def canny(
    input: Tensor,
    low_threshold: float = 0.01,
    high_threshold: float = 0.1,
    kernel_size: Union[Tuple[int, int] , int]= (3, 3),
    sigma: Union[Tuple[float, float] , Tensor] = (1, 1),
    hysteresis: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    r"""Find edges of the input image and filters them using the Canny algorithm.

    .. image:: _static/img/canny.png

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/canny.html>`__.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = canny(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """

    low_threshold = (input.max() * low_threshold).item()
    high_threshold = (input.max() * high_threshold).item()

    device = input.device
    dtype = input.dtype
    
    # plt.figure()
    # plt.imshow(((input.squeeze().permute(1, 2, 0).cpu().numpy()) * 255).astype('uint8'))
    # plt.title("Original")

    # To Grayscale
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)

    # plt.figure()
    # plt.imshow(input.squeeze().cpu().numpy())
    # plt.title("GrayScale")

    # Gaussian filter
    blurred: Tensor = gaussian_blur2d(input, kernel_size, sigma)

    # plt.figure()
    # plt.imshow(blurred.squeeze().cpu().numpy())
    # plt.title("Blurred")

    # Compute the gradients
    gradients: Tensor = spatial_gradient(blurred, normalized=True)

    # Unpack the edges
    gx: Tensor = gradients[:, :, 0]
    gy: Tensor = gradients[:, :, 1]

    # Compute gradient magnitude and angle
    magnitude: Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    angle: Tensor = torch.atan2(gy, gx)

    # plt.figure()
    # plt.imshow(magnitude.squeeze().cpu().numpy())
    # plt.title("Magnitude")

    # Radians to Degrees
    angle = 180.0 * angle / math.pi

    # plt.figure()
    # plt.imshow(angle.squeeze().cpu().numpy())
    # plt.title("Angle")

    # Round angle to the nearest 45 degree
    angle = torch.round(angle / 45) * 45

    # Non-maximal suppression
    nms_kernels: Tensor = get_canny_nms_kernel(device, dtype)
    nms_magnitude: Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

    # plt.figure()
    # plt.imshow(nms_magnitude.squeeze().cpu().numpy())

    # Get the indices for both directions
    positive_idx: Tensor = (angle / 45) % 8
    positive_idx = positive_idx.long()

    negative_idx: Tensor = ((angle / 45) + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive: Tensor = torch.gather(nms_magnitude, 1, positive_idx)
    channel_select_filtered_negative: Tensor = torch.gather(nms_magnitude, 1, negative_idx)

    channel_select_filtered: Tensor = torch.stack(
        [channel_select_filtered_positive, channel_select_filtered_negative], 1
    )

    is_max: Tensor = channel_select_filtered.min(dim=1)[0] > 0.0
    is_max = is_max.to(dtype)
    magnitude = magnitude * is_max

    # plt.figure()
    # plt.imshow(magnitude.squeeze().cpu().numpy())
    # plt.title("Magnitude after NMS")
    
    # Threshold
    low: Tensor = F.threshold(magnitude, low_threshold, 0.0)
    high: Tensor = F.threshold(magnitude, high_threshold, 0.0)

    # plt.figure()
    # plt.imshow(edges.squeeze().cpu().numpy())
    # plt.title("Magnitude after Threshold")
    # print(magnitude.max(), magnitude.min())

    # low: Tensor = magnitude > low_threshold
    # high: Tensor = magnitude > high_threshold

    # low = low.to(dtype)
    # high = high.to(dtype)

    edges = low * 0.5 + high * 0.5
    # edges = edges.to(dtype)

    # plt.figure()
    # plt.imshow(edges.squeeze().cpu().numpy())
    # plt.title("Edges")
    # plt.figure()
    # plt.imshow(high.squeeze().cpu().numpy())
    # plt.title("High")
    # Hysteresis
    if hysteresis:
        edges_old: Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
        hysteresis_kernels: Tensor = get_hysteresis_kernel(device, dtype)
        # changed values here
        while ((edges_old - edges).abs() != 0).any():
            weak: Tensor = (edges == 0.5).float()
            strong: Tensor = (edges == 1).float()

            hysteresis_magnitude: Tensor = F.conv2d(
                edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2
            )
            hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
            hysteresis_magnitude = hysteresis_magnitude * weak + strong

            edges_old = edges.clone()
            edges = hysteresis_magnitude + (hysteresis_magnitude == 0).to(dtype) * weak * 0.5

        edges = hysteresis_magnitude

    # return magnitude, edges
    return edges


class Canny(Module):
    r"""Module that finds edges of the input image and filters them using the Canny algorithm.

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.4,
        kernel_size: Union[Tuple[int, int] , int] = (5, 5),
        sigma: Union[Tuple[float, float] , Tensor] = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Double threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Hysteresis
        self.hysteresis = hysteresis

        self.eps: float = eps

    def __repr__(self) -> str:
        return "".join(
            (
                f"{type(self).__name__}(",
                ", ".join(
                    f"{name}={getattr(self, name)}" for name in sorted(self.__dict__) if not name.startswith("_")
                ),
                ")",
            )
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        return canny(
            input, self.low_threshold, self.high_threshold, self.kernel_size, self.sigma, self.hysteresis, self.eps
        )


if __name__ == '__main__':
    import torch

    x = torch.randn([1, 3, 224, 224], device='cuda')
    x = (x - x.min())/(x.max() - x.min())
    print(x.device)
    m = Canny() 
    y = m(x)
    print(y[0].shape, y[1].shape)