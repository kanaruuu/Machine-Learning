from uuid import UUID

from sqlalchemy import insert

from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from ..core.IAddRequestUploadVideoFunction import IAddRequestUploadVideoFunction
from ...models.RequestUploadVideosTable import RequestUploadVideosTable


class AddRequestUploadVideoFunction(IAddRequestUploadVideoFunction):

    def __init__(
        self,
        database_session_factory: ISessionFactory
    ):
        self.__database_session_factory = database_session_factory

    async def __call__(
        self,
        request_id: str,
        filekey: str
    ):
        async with self.__database_session_factory() as session:
            query = (
                insert(RequestUploadVideosTable)
                .values(
                    request_id=request_id,
                    filekey=filekey,
                    is_complete=False
                )
            )
            await session.execute(query)
            await session.commit()