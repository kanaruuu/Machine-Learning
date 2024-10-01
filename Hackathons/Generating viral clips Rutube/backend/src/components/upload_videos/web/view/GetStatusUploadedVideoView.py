from components.upload_videos.functions.core.IGetRequestUploadStatusFunction import IGetRequestUploadStatusFunction


class GetStatusUploadedVideoView:

    def __init__(
        self,
        get_request_upload_status_function: IGetRequestUploadStatusFunction
    ):
        self.__get_request_upload_status_function = get_request_upload_status_function

    async def __call__(self, request_id: str):
        is_complete = await self.__get_request_upload_status_function(
            request_id=request_id
        )
        return {
            "request_id": request_id,
            "is_complete": is_complete
        }