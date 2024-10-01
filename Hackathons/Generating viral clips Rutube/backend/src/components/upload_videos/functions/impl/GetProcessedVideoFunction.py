from typing import List

from sqlalchemy import select

from components.separator_video.models.FinishedProcessedRequestTable import FinishedProcessedRequestTable
from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from ..core.IGetProcessedVideoFunction import IGetProcessedVideoFunction


class GetProcessedVideoFunction(IGetProcessedVideoFunction):

    def __init__(
        self,
        database_session_factory: ISessionFactory
    ):
        self.__database_session_factory = database_session_factory

    async def __call__(self, request_id: str) -> List[FinishedProcessedRequestTable]:
        async with self.__database_session_factory() as session:
            query = (
                select(FinishedProcessedRequestTable)
                .where(FinishedProcessedRequestTable.request_id == request_id)
            )
            result = await session.execute(query)
            processed_video_list = result.scalars().all()

        return processed_video_list