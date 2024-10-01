from sqlalchemy import insert

from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from ..core.IAddProcessedVideosFunction import IAddProcessedVideosFunction
from ...models.FinishedProcessedRequestTable import FinishedProcessedRequestTable


class AddProcessedVideosFunction(IAddProcessedVideosFunction):

    def __init__(
        self,
        database_session_factory: ISessionFactory
    ):
        self.__database_session_factory = database_session_factory

    async def __call__(
        self,
        request_id: str,
        filekey: str,
        metrics: str,
        metrics_fields: str
    ):
        async with self.__database_session_factory() as session:
            query = (
                insert(FinishedProcessedRequestTable)
                .values(
                    request_id=request_id,
                    filekey=filekey,
                    metrics=metrics,
                    metrics_fields=metrics_fields
                )
            )
            await session.execute(query)
            await session.commit()