from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

class session_maker_custom(sessionmaker):
    async def __aenter__(self) -> AsyncSession:
        pass

class ISessionFactory:

    def __call__(self) -> session_maker_custom:
        raise NotImplementedError()
