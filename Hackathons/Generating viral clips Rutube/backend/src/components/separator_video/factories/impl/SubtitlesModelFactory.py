from ..core.ISubtitlesModelFactory import ISubtitlesModelFactory


class SubtitlesModelFactory(ISubtitlesModelFactory):

    def __init__(
        self,
        model
    ):
        self.__model = model

    def __call__(self):
        return self.__model