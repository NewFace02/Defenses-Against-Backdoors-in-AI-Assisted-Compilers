from abc import ABC, abstractmethod

from Reproduction.StegoUnified.src.common import BuildConfig, BuildResult


class BaseBuilder(ABC):
    @abstractmethod
    def build(self, config: BuildConfig) -> BuildResult:
        raise NotImplementedError