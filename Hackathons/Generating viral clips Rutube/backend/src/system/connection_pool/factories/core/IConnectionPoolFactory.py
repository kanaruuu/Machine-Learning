from asyncpg import Pool


class IConnectionPoolFactory:

    def __call__(self) -> Pool:
        raise NotImplementedError()