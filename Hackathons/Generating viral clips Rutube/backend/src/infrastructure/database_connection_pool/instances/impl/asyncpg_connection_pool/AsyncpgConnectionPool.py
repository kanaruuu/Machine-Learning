from asyncpg import Pool

from system.connection_pool.instances.core.IConnectionPool import IConnectionPool


class AsyncpgConnectionPool(IConnectionPool):

    def __init__(
        self,
        connection_pool: Pool
    ):
        self.__connection_pool = connection_pool

    def acquire(self):
        return self.__connection_pool.acquire()
