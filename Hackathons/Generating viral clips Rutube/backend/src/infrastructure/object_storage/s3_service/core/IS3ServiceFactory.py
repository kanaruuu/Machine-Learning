from ..core.IS3Client import IS3Client


class IS3ServiceFactory:

    def __call__(self) -> IS3Client:
        raise NotImplementedError()