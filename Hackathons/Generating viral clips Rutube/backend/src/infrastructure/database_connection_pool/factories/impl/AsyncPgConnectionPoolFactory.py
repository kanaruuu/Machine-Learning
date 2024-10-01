import asyncpg
from asyncpg import Pool

from .models.AsyncPgConfig import AsyncPgConfig
from ..core.IConnectionPoolFactory import IConnectionPoolFactory
from ...instances.impl.asyncpg_connection_pool.AsyncpgConnectionPool import AsyncpgConnectionPool


class AsyncPgConnectionPoolFactory(IConnectionPoolFactory):

    def __init__(
        self,
        config: AsyncPgConfig
    ):
        self.__config = config

    def __call__(self) -> Pool:
        asyncpg_create_pool = asyncpg.create_pool(
            host=self.__config.host,
            port=self.__config.port,
            database=self.__config.database,
            user=self.__config.user,
            password=self.__config.password,
            min_size=self.__config.min_size,
            max_size=self.__config.max_size
        )
        return asyncpg_create_pool
        # return AsyncpgConnectionPool(
        #     connection_pool=asyncpg_create_pool
        # )
