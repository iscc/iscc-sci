import pytest
import iscc_sci.utils

CHECKSUM = "9db4c0d9e68c5203dc8c2fefe52fa5d54671be3a3253e06888cace7c60e5a743"


def test_check_integrity_success(img_path):
    """Test the function with a matching checksum."""
    result = iscc_sci.utils.check_integrity(img_path, CHECKSUM)
    assert result == img_path


def test_check_integrity_failure(img_path):
    """Test the function with a non-matching checksum."""
    wrong_checksum = CHECKSUM[:-4]
    with pytest.raises(RuntimeError) as exc_info:
        iscc_sci.utils.check_integrity(img_path, wrong_checksum)
    assert "Failed integrity check" in str(exc_info.value)
