from abc import ABC, abstractmethod

class Watermarking(ABC):
    def __init__(self):
        pass 

    @abstractmethod
    def Enc(self, I, W, K):
        pass

    @abstractmethod
    def Dec(self, D, I, K):
        pass
