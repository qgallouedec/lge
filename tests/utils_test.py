import numpy as np
import pytest
import torch
from gym import spaces

from lge.utils import get_size, index, indexes, round, sample_geometric_with_max, preprocess


def test_indexes():
    a = np.array([3, 4])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4], [5, 4]])
    assert (indexes(a, b) == np.array([3, 4])).all()


def test_indexes_when_none():
    a = np.array([-1, -1])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4], [5, 4]])
    assert (indexes(a, b) == np.array([])).all()


def test_index():
    a = np.array([3, 4])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4]])
    assert index(a, b) == 3


def test_index_when_none():
    a = np.array([-1, -1])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4]])
    assert index(a, b) is None


@pytest.mark.parametrize("mean", [2.0, 4.0])
@pytest.mark.parametrize("max_value", [4, 5])
@pytest.mark.parametrize("size", [None, 4, (3, 5)])
def test_sample_geometric_with_max(mean, max_value, size):
    p = 1 / mean
    sample = [sample_geometric_with_max(p, max_value, size) for _ in range(1000)]
    if size is not None:
        assert sample[0].shape == (size if type(size) is tuple else (size,))
        assert sample[0].dtype == int
    else:
        assert type(sample[0]) is int
    _, counts = np.unique(sample, return_counts=True)
    sampled_dist = counts / counts.sum()
    true_weights = np.array([(1 - p) ** (k - 1) * p for k in range(1, max_value + 1)])
    true_dist = true_weights / true_weights.sum()
    assert np.isclose(sampled_dist, true_dist, atol=0.05).all()


def test_round():
    x = torch.Tensor([0.0, 0.4, 0.8, 1.2, 1.6])
    y = round(x, decimals=0.2)
    z = torch.Tensor([0.0000, 0.6310, 0.6310, 1.2619, 1.8929])
    torch.isclose(y, z, atol=0.0001)


def test_get_size():
    assert get_size(spaces.Discrete(3)) == 3
    assert get_size(spaces.MultiDiscrete([3, 2])) == 5
    assert get_size(spaces.MultiBinary(3)) == 3
    # assert get_size(spaces.MultiBinary([3, 2])) == 5
    assert get_size(spaces.Box(-2, 2, shape=(2,))) == 2
    assert get_size(spaces.Box(-2, 2, shape=(2, 2))) == 4


def test_preprocess_discrete():
    actual = preprocess(torch.tensor(2, dtype=torch.long), spaces.Discrete(3))
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_batched_discrete():
    actual = preprocess(torch.tensor([2, 1], dtype=torch.long), spaces.Discrete(3))
    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_multidiscrete():
    actual = preprocess(torch.tensor([2, 0], dtype=torch.long), spaces.MultiDiscrete([3, 2]))
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_batched_multidiscrete():
    actual = preprocess(torch.tensor([[2, 0], [1, 1]], dtype=torch.long), spaces.MultiDiscrete([3, 2]))
    expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_multibinary():
    actual = preprocess(torch.tensor([1, 0, 1], dtype=torch.long), spaces.MultiBinary(3))
    expected = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_bached_multibinary():
    actual = preprocess(torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.long), spaces.MultiBinary(3))
    expected = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_multidimensional_multibinary():
    actual = preprocess(torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.long), spaces.MultiBinary([3, 2]))
    expected = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_batched_multidimensional_multibinary():
    actual = preprocess(
        torch.tensor([[[1, 0], [1, 1], [0, 1]], [[0, 0], [0, 1], [1, 0]]], dtype=torch.long), spaces.MultiBinary([3, 2])
    )
    expected = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_preprocess_box():
    input = torch.tensor([0.7, -0.2, 0.2, 0.6], dtype=torch.float32)
    actual = preprocess(input, spaces.Box(-2, 2, shape=(4,)))
    expected = input
    torch.testing.assert_close(actual, expected)


def test_preprocess_batched_box():
    input = torch.tensor([[-0.5, -0.6, 0.9, -0.4], [-0.4, -0.4, -0.2, -0.4]], dtype=torch.float32)
    actual = preprocess(input, spaces.Box(-2, 2, shape=(4,)))
    expected = input
    torch.testing.assert_close(actual, expected)


def test_preprocess_multidimensional_box():
    input = torch.tensor(
        [[1.2, -0.3, -1.9, -0.1], [-0.3, -0.0, 0.0, -0.0], [0.8, 1.8, -1.3, 1.2]],
        dtype=torch.float32,
    )
    actual = preprocess(input, spaces.Box(-2, 2, shape=(3, 4)))
    expected = input
    torch.testing.assert_close(actual, expected)


def test_preprocess_batched_multidimensional_box():
    input = torch.tensor(
        [
            [[1.6, -0.7, 0.8, -0.4], [-0.5, -1.2, 0.0, -0.0], [-0.1, 0.4, -1.2, 1.1]],
            [[0.0, 0.5, -1.2, -0.3], [-0.8, 1.7, 1.0, 0.1], [1.3, 0.1, -0.2, 0.6]],
        ],
        dtype=torch.float32,
    )
    actual = preprocess(input, spaces.Box(-2, 2, shape=(3, 4)))
    expected = input
    torch.testing.assert_close(actual, expected)


def test_preprocess_image_channel_last():
    input = torch.tensor(
        [
            [[43, 230, 206], [221, 193, 212], [79, 123, 167], [180, 88, 55]],
            [[50, 30, 17], [95, 159, 241], [76, 37, 39], [167, 89, 65]],
            [[221, 35, 147], [246, 149, 216], [233, 90, 100], [75, 3, 16]],
            [[195, 129, 49], [73, 61, 135], [202, 82, 62], [133, 231, 85]],
        ],
        dtype=torch.uint8,
    )
    actual = preprocess(input, spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8))
    expected = torch.tensor(
        [
            [
                [0.16862745, 0.86666667, 0.30980392, 0.70588235],
                [0.19607843, 0.37254902, 0.29803922, 0.65490196],
                [0.86666667, 0.96470588, 0.91372549, 0.29411765],
                [0.76470588, 0.28627451, 0.79215686, 0.52156863],
            ],
            [
                [0.90196078, 0.75686275, 0.48235294, 0.34509804],
                [0.11764706, 0.62352941, 0.14509804, 0.34901961],
                [0.1372549, 0.58431373, 0.35294118, 0.01176471],
                [0.50588235, 0.23921569, 0.32156863, 0.90588235],
            ],
            [
                [0.80784314, 0.83137255, 0.65490196, 0.21568627],
                [0.06666667, 0.94509804, 0.15294118, 0.25490196],
                [0.57647059, 0.84705882, 0.39215686, 0.0627451],
                [0.19215686, 0.52941176, 0.24313725, 0.33333333],
            ],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_preprocess_image_channel_first():
    input = torch.tensor(
        [
            [[244, 18, 35, 248], [219, 203, 190, 198], [38, 1, 216, 148], [231, 188, 154, 49]],
            [[15, 135, 66, 236], [242, 157, 20, 80], [222, 92, 171, 101], [25, 66, 191, 65]],
            [[78, 16, 139, 73], [92, 2, 227, 188], [86, 147, 190, 103], [45, 69, 125, 254]],
        ],
        dtype=torch.uint8,
    )
    actual = preprocess(input, spaces.Box(0, 255, shape=(3, 4, 4), dtype=np.uint8))
    expected = torch.tensor(
        [
            [
                [0.95686275, 0.07058824, 0.1372549, 0.97254902],
                [0.85882353, 0.79607843, 0.74509804, 0.77647059],
                [0.14901961, 0.00392157, 0.84705882, 0.58039216],
                [0.90588235, 0.7372549, 0.60392157, 0.19215686],
            ],
            [
                [0.05882353, 0.52941176, 0.25882353, 0.9254902],
                [0.94901961, 0.61568627, 0.07843137, 0.31372549],
                [0.87058824, 0.36078431, 0.67058824, 0.39607843],
                [0.09803922, 0.25882353, 0.74901961, 0.25490196],
            ],
            [
                [0.30588235, 0.0627451, 0.54509804, 0.28627451],
                [0.36078431, 0.00784314, 0.89019608, 0.7372549],
                [0.3372549, 0.57647059, 0.74509804, 0.40392157],
                [0.17647059, 0.27058824, 0.49019608, 0.99607843],
            ],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_preprocess_batched_image_channel_last():
    input = torch.tensor(
        [
            [
                [[24, 55, 52], [77, 147, 2], [169, 234, 114], [193, 84, 91]],
                [[180, 127, 235], [197, 65, 137], [36, 0, 128], [198, 69, 132]],
                [[243, 55, 185], [134, 203, 41], [173, 133, 155], [202, 47, 59]],
                [[174, 127, 31], [60, 173, 53], [109, 186, 157], [233, 231, 254]],
            ],
            [
                [[239, 15, 161], [93, 37, 116], [178, 231, 35], [190, 78, 68]],
                [[53, 31, 203], [129, 17, 77], [66, 254, 118], [246, 220, 4]],
                [[197, 233, 44], [119, 84, 96], [214, 238, 171], [24, 164, 67]],
                [[16, 85, 136], [243, 136, 57], [38, 23, 55], [52, 91, 66]],
            ],
        ],
        dtype=torch.uint8,
    )
    actual = preprocess(input, spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8))
    expected = torch.tensor(
        [
            [
                [
                    [0.09411765, 0.30196078, 0.6627451, 0.75686275],
                    [0.70588235, 0.77254902, 0.14117647, 0.77647059],
                    [0.95294118, 0.5254902, 0.67843137, 0.79215686],
                    [0.68235294, 0.23529412, 0.42745098, 0.91372549],
                ],
                [
                    [0.21568627, 0.57647059, 0.91764706, 0.32941176],
                    [0.49803922, 0.25490196, 0.0, 0.27058824],
                    [0.21568627, 0.79607843, 0.52156863, 0.18431373],
                    [0.49803922, 0.67843137, 0.72941176, 0.90588235],
                ],
                [
                    [0.20392157, 0.00784314, 0.44705882, 0.35686275],
                    [0.92156863, 0.5372549, 0.50196078, 0.51764706],
                    [0.7254902, 0.16078431, 0.60784314, 0.23137255],
                    [0.12156863, 0.20784314, 0.61568627, 0.99607843],
                ],
            ],
            [
                [
                    [0.9372549, 0.36470588, 0.69803922, 0.74509804],
                    [0.20784314, 0.50588235, 0.25882353, 0.96470588],
                    [0.77254902, 0.46666667, 0.83921569, 0.09411765],
                    [0.0627451, 0.95294118, 0.14901961, 0.20392157],
                ],
                [
                    [0.05882353, 0.14509804, 0.90588235, 0.30588235],
                    [0.12156863, 0.06666667, 0.99607843, 0.8627451],
                    [0.91372549, 0.32941176, 0.93333333, 0.64313725],
                    [0.33333333, 0.53333333, 0.09019608, 0.35686275],
                ],
                [
                    [0.63137255, 0.45490196, 0.1372549, 0.26666667],
                    [0.79607843, 0.30196078, 0.4627451, 0.01568627],
                    [0.17254902, 0.37647059, 0.67058824, 0.2627451],
                    [0.53333333, 0.22352941, 0.21568627, 0.25882353],
                ],
            ],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


def test_preprocess_batched_image_channel_first():
    input = torch.tensor(
        [
            [
                [[245, 119, 79, 4], [111, 29, 172, 243], [17, 211, 213, 7], [40, 16, 31, 178]],
                [[25, 69, 164, 187], [92, 148, 125, 200], [160, 118, 238, 200], [7, 144, 204, 96]],
                [[164, 79, 87, 98], [136, 11, 70, 216], [13, 250, 109, 154], [51, 156, 36, 162]],
            ],
            [
                [[138, 124, 127, 11], [76, 201, 5, 62], [248, 161, 22, 246], [12, 241, 137, 237]],
                [[213, 28, 143, 77], [23, 28, 55, 54], [49, 29, 71, 192], [195, 211, 130, 191]],
                [[22, 254, 214, 234], [222, 144, 27, 54], [219, 167, 123, 199], [38, 4, 87, 126]],
            ],
        ],
        dtype=torch.uint8,
    )
    actual = preprocess(input, spaces.Box(0, 255, shape=(3, 4, 4), dtype=np.uint8))
    expected = torch.tensor(
        [
            [
                [
                    [0.96078431, 0.46666667, 0.30980392, 0.01568627],
                    [0.43529412, 0.11372549, 0.6745098, 0.95294118],
                    [0.06666667, 0.82745098, 0.83529412, 0.02745098],
                    [0.15686275, 0.0627451, 0.12156863, 0.69803922],
                ],
                [
                    [0.09803922, 0.27058824, 0.64313725, 0.73333333],
                    [0.36078431, 0.58039216, 0.49019608, 0.78431373],
                    [0.62745098, 0.4627451, 0.93333333, 0.78431373],
                    [0.02745098, 0.56470588, 0.8, 0.37647059],
                ],
                [
                    [0.64313725, 0.30980392, 0.34117647, 0.38431373],
                    [0.53333333, 0.04313725, 0.2745098, 0.84705882],
                    [0.05098039, 0.98039216, 0.42745098, 0.60392157],
                    [0.2, 0.61176471, 0.14117647, 0.63529412],
                ],
            ],
            [
                [
                    [0.54117647, 0.48627451, 0.49803922, 0.04313725],
                    [0.29803922, 0.78823529, 0.01960784, 0.24313725],
                    [0.97254902, 0.63137255, 0.08627451, 0.96470588],
                    [0.04705882, 0.94509804, 0.5372549, 0.92941176],
                ],
                [
                    [0.83529412, 0.10980392, 0.56078431, 0.30196078],
                    [0.09019608, 0.10980392, 0.21568627, 0.21176471],
                    [0.19215686, 0.11372549, 0.27843137, 0.75294118],
                    [0.76470588, 0.82745098, 0.50980392, 0.74901961],
                ],
                [
                    [0.08627451, 0.99607843, 0.83921569, 0.91764706],
                    [0.87058824, 0.56470588, 0.10588235, 0.21176471],
                    [0.85882353, 0.65490196, 0.48235294, 0.78039216],
                    [0.14901961, 0.01568627, 0.34117647, 0.49411765],
                ],
            ],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)
