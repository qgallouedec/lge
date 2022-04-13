def compute_output(input_size, kernel_size, stride, padding):
    a = input_size - kernel_size + 2 * padding
    assert a % stride == 0, a % stride
    return (input_size - kernel_size + 2 * padding) // stride + 1


def compute_output_t(input_size, kernel_size, stride, padding, output_padding=0):
    return (input_size - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1


if __name__ == "__main__":
    size = compute_output(input_size=84, kernel_size=6, stride=2, padding=1)
    print(size)
    size = compute_output(input_size=size, kernel_size=5, stride=2, padding=1)
    print(size)
    size = compute_output(input_size=size, kernel_size=6, stride=2, padding=1)
    print(size)
    size = compute_output(input_size=size, kernel_size=5, stride=2, padding=1)
    print(size)
