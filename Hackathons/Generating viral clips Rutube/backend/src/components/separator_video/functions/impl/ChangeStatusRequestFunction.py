from sqlalchemy import update

from components.upload_videos.models.RequestUploadVideosTable import RequestUploadVideosTable
from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from ..core.IChangeStatusRequestFunction import IChangeStatusRequestFunction


class ChangeStatusRequestFunction(IChangeStatusRequestFunction):

    def __init__(
        self,
        database_session_factory: ISessionFactory
    ):
        self.__database_session_factory = database_session_factory

    async def __call__(self, request_id: str):
        async with self.__database_session_factory() as session:
            query = (
                update(RequestUploadVideosTable)
                .where(RequestUploadVideosTable.request_id == request_id)
                .values(
                    is_complete=True
                )
            )
            await session.execute(query)
            await session.commit()