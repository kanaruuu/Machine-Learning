class IAddProcessedVideosFunction:

    async def __call__(
        self,
        request_id: str,
        filekey: str,
        metrics: str,
        metrics_fields: str
    ):
        raise NotImplementedError()