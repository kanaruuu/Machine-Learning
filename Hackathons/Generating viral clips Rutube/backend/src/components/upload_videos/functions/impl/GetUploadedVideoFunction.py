from typing import List

from sqlalchemy import select

from ..core.IGetUploadedVideoFunction import IGetUploadedVideoFunction

from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from ...models.RequestUploadVideosTable import RequestUploadVideosTable


class GetUploadedVideoFunction(IGetUploadedVideoFunction):

    def __init__(
        self,
        database_session_factory: ISessionFactory
    ):
        self.__database_session_factory = database_session_factory

    async def __call__(
        self
    ) -> List[RequestUploadVideosTable]:
        async with self.__database_session_factory() as session:
            query = (
                select(RequestUploadVideosTable)
            )
            result = await session.execute(query)
            video_list = result.scalars().all()

        return video_list