from sqlalchemy import UUID


class IGetRequestUploadStatusFunction:

    async def __call__(self, request_id: str) -> bool:
        raise NotImplementedError()