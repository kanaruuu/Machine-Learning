from typing import TypeVar, Generic

from aio_pika import IncomingMessage

from ...models.BaseModelMessage import BaseModelMessage

E = TypeVar('E', bound=BaseModelMessage)

class IEventHandler(Generic[E]):

    async def __call__(self, event: E, message: IncomingMessage):
        raise NotImplementedError()