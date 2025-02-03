"""Type definitions for diarized_transcriber."""

from typing import Union, TypeVar, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    AudioArray = npt.NDArray[np.float32]
else:
    AudioArray = np.ndarray

# Type variable for generic numpy arrays
ArrayType = TypeVar('ArrayType', bound=np.ndarray)