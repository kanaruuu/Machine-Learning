from sqlalchemy.orm import sessionmaker

from infrastructure.database_connection_pool.factories.core.ISessionFactory import session_maker_custom, ISessionFactory


class SessionFactory(ISessionFactory):

    def __init__(
        self,
        session_maker: sessionmaker
    ):
        self.__session_maker = session_maker

    def __call__(self) -> session_maker_custom:
        return self.__session_maker()
