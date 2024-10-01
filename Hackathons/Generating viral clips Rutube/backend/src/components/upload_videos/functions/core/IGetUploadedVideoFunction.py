from typing import List

from components.upload_videos.models.RequestUploadVideosTable import RequestUploadVideosTable


class IGetUploadedVideoFunction:

    async def __call__(self) -> List[RequestUploadVideosTable]:
        raise NotImplementedError()