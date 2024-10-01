class IChangeStatusRequestFunction:

    async def __call__(self, request_id: str):
        raise NotImplementedError()