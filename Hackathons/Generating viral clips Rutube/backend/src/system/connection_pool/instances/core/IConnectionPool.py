class IConnectionPool:

    def acquire(self):
        raise NotImplementedError()