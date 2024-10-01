from sqlalchemy import UUID, and_, select

from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from ..core.IGetRequestUploadStatusFunction import IGetRequestUploadStatusFunction
from ...models.RequestUploadVideosTable import RequestUploadVideosTable


class GetRequestUploadStatusFunction(IGetRequestUploadStatusFunction):

    def __init__(
        self,
        database_session_factory: ISessionFactory
    ):
        self.__database_session_factory = database_session_factory

    async def __call__(self, request_id: str) -> bool:
        async with self.__database_session_factory() as session:
            query = (
                select(RequestUploadVideosTable)
                .where(and_(
                    RequestUploadVideosTable.request_id == request_id,
                    RequestUploadVideosTable.is_complete == True
                ))
            )
            result = await session.execute(query)
            request_upload = result.fetchone()

        if request_upload:
            return True
        else:
            return False