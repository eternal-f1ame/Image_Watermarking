"""Abstract class for watermarking algorithms."""
from abc import ABC, abstractmethod

class Watermarking(ABC):
    """Abstract class for watermarking algorithms."""

    def __init__(self):
        pass

    @abstractmethod
    def enc(self, _i, _w, _k):
        """Encrypts an image _i with a watermark _w and a key _k."""
        pass

    @abstractmethod
    def dec(self, _d, _i, _k):
        """Decrypts an image _d with a key _k."""
        pass
