from components.upload_videos.functions.core.IGetUploadedVideoFunction import IGetUploadedVideoFunction


class GetUploadedVideoView:

    def __init__(
        self,
        get_uploaded_video_function: IGetUploadedVideoFunction
    ):
        self.__get_uploaded_video_function = get_uploaded_video_function

    async def __call__(self):
        video_list = await self.__get_uploaded_video_function()

        return [{
            "request_id": str(video.request_id),
            "filekey": video.filekey,
            "is_complete": video.is_complete
        } for video in video_list]