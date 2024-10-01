from sqlalchemy.orm import sessionmaker

from system.connection_pool.factories.core.ISessionFactory import ISessionFactory, session_maker_custom

class SessionFactory(ISessionFactory):

    def __init__(
        self,
        session_maker: sessionmaker
    ):
        self.__session_maker = session_maker

    def __call__(self) -> session_maker_custom:
        return self.__session_maker()
