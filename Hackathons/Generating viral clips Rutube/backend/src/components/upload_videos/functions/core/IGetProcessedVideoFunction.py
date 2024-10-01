from typing import List

from components.separator_video.models.FinishedProcessedRequestTable import FinishedProcessedRequestTable


class IGetProcessedVideoFunction:

    async def __call__(self, request_id: str) -> List[FinishedProcessedRequestTable]:
        raise NotImplementedError()