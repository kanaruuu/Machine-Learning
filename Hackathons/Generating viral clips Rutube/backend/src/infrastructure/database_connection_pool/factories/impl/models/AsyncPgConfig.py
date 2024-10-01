from typing import Optional

from pydantic import BaseModel


class AsyncPgConfig(BaseModel):
    host: str
    port: str
    database: str
    user: str
    password: str
    min_size: Optional[int] = 15
    max_size: Optional[int] = 15