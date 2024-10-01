from asyncpg import Pool

from system.connection_pool.instances.core.IConnectionPool import IConnectionPool


class IConnectionPoolFactory:

    def __call__(self) -> Pool:
        raise NotImplementedError()