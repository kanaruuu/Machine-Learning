from components.upload_videos.functions.core.IGetProcessedVideoFunction import IGetProcessedVideoFunction


class GetProcessedVideoView:

    def __init__(
        self,
        get_processed_video_function: IGetProcessedVideoFunction
    ):
        self.__get_processed_video_function = get_processed_video_function

    async def __call__(self, request_id: str):
        link_video_list = await self.__get_processed_video_function(
            request_id=request_id
        )

        return [{
            "request_id": str(link_video.request_id),
            "video_url": f"https://06f9ff45-22a4-4d07-87c5-2a0e4ba9c4d1.selstorage.ru/{link_video.filekey}",
            "metrics": link_video.metrics,
            "metrics_fields": link_video.metrics_fields
        } for link_video in link_video_list]

