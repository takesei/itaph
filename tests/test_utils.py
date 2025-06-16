import numpy as np
import pytest

from itaph.utils import convert_to_affine


@pytest.mark.smoke
class TestAffineConverter:
    def test_translation_only(self):
        b = np.array([1.0, 2.0, -3.0])
        M = convert_to_affine(A=None, b=b)
        expected = np.eye(4)
        expected[:3, 3] = b
        assert np.allclose(M, expected)

    def test_linear_only(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        M = convert_to_affine(A=A, b=None)
        expected = np.eye(3)
        expected[:2, :2] = A
        assert np.allclose(M, expected)

    def test_linear_and_translation(self):
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        b = np.array([5.0, -5.0])
        M = convert_to_affine(A=A, b=b)
        expected = np.eye(3)
        expected[:2, :2] = A
        expected[:2, 2] = b
        assert np.allclose(M, expected)

    @pytest.mark.parametrize(
        'A,b,expectation',
        [
            (
                None,
                None,
                pytest.raises(
                    ValueError, match='At least one of A or b must be specified'
                ),
            ),
            (
                np.zeros((2, 3)),
                None,
                pytest.raises(ValueError, match='A must be a square matrix'),
            ),
            (
                None,
                np.zeros((2, 2)),
                pytest.raises(ValueError, match='b must be a 1D array of length'),
            ),
        ],
    )
    def test_invalid_inputs(self, A, b, expectation):
        with expectation:
            convert_to_affine(A=A, b=b)
