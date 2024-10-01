class IAddRequestUploadVideoFunction:

    async def __call__(
        self,
        request_id: str,
        filekey: str
    ):
        raise NotImplementedError()
