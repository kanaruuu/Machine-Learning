from fastapi import FastAPI
from starlette import status

from infrastructure.amqp_messaging.common.core.IRabbitFactory import IRabbitFactory
from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from infrastructure.object_storage.s3_service.core.IS3ServiceFactory import IS3ServiceFactory
from system.ioc.ioc import ioc
from .view.CreateUploadVideoView import CreateUploadVideoView
from .view.GetProcessedVideoView import GetProcessedVideoView
from .view.GetStatusUploadedVideoView import GetStatusUploadedVideoView
from .view.GetUploadedVideoView import GetUploadedVideoView
from ..functions.impl.AddRequestUploadVideoFunction import AddRequestUploadVideoFunction
from ..functions.impl.GetProcessedVideoFunction import GetProcessedVideoFunction
from ..functions.impl.GetRequestUploadStatusFunction import GetRequestUploadStatusFunction
from ..functions.impl.GetUploadedVideoFunction import GetUploadedVideoFunction


class UploadVideoWebInstall:

    def __call__(
        self,
        app: FastAPI
    ):
        create_upload_video_view = CreateUploadVideoView(
            s3_service_factory=ioc.get(IS3ServiceFactory),
            amqp_messaging_factory=ioc.get(IRabbitFactory),
            add_request_upload_video_function=AddRequestUploadVideoFunction(
                database_session_factory=ioc.get(ISessionFactory)
            )
        )

        get_status_uploaded_video_view = GetStatusUploadedVideoView(
            get_request_upload_status_function=GetRequestUploadStatusFunction(
                database_session_factory=ioc.get(ISessionFactory)
            )
        )

        get_processed_video_view = GetProcessedVideoView(
            get_processed_video_function=GetProcessedVideoFunction(
                database_session_factory=ioc.get(ISessionFactory)
            )
        )

        get_uploaded_video = GetUploadedVideoView(
            get_uploaded_video_function=GetUploadedVideoFunction(
                database_session_factory=ioc.get(ISessionFactory)
            )
        )

        app.add_api_route(
            path="/video/upload",
            endpoint=create_upload_video_view.__call__,
            methods=["POST"],
            status_code=status.HTTP_201_CREATED,
            tags=["video"]
        )

        app.add_api_route(
            path="/video/status",
            endpoint=get_status_uploaded_video_view.__call__,
            methods=["GET"],
            tags=["video"]
        )

        app.add_api_route(
            path="/video/processed/list",
            endpoint=get_processed_video_view.__call__,
            methods=["GET"],
            tags=["video"]
        )

        app.add_api_route(
            path="/video/list",
            endpoint=get_uploaded_video.__call__,
            methods=["GET"],
            tags=["video"]
        )
