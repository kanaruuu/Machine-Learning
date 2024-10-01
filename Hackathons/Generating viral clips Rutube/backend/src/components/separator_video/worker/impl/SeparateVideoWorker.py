from components.separator_video.factories.core.ISubtitlesModelFactory import ISubtitlesModelFactory
from components.separator_video.functions.impl.AddProcessedVideosFunction import AddProcessedVideosFunction
from components.separator_video.functions.impl.ChangeStatusRequestFunction import ChangeStatusRequestFunction
from components.separator_video.handlers.SeparateVideoHandler import SeparateVideoHandler
from components.upload_videos.domain.models.StartSeparateVideoMessageModel import StartSeparateVideoMessageModel
from infrastructure.amqp_messaging.common.core.IRabbitFactory import IRabbitFactory
from infrastructure.amqp_messaging.common.core.IRabbitMessaging import IRabbitMessaging
from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from infrastructure.object_storage.s3_service.core.IS3ServiceFactory import IS3ServiceFactory
from system.ioc.ioc import ioc


class SeparateVideoWorker:

    def __init__(
        self,
        rabbitmq_messaging_factory: IRabbitFactory
    ):
        self.__rabbitmq_messaging_factory = rabbitmq_messaging_factory

    async def start(
        self
    ):
        rabbitmq_messaging: IRabbitMessaging = await self.__rabbitmq_messaging_factory()

        booking_repeat_event = SeparateVideoHandler(
            s3_service_factory=ioc.get(IS3ServiceFactory),
            add_processed_videos_function=AddProcessedVideosFunction(
                database_session_factory=ioc.get(ISessionFactory)
            ),
            subtitles_model_factory=ioc.get(ISubtitlesModelFactory),
            change_status_request_function=ChangeStatusRequestFunction(
                database_session_factory=ioc.get(ISessionFactory)
            )
        )

        await rabbitmq_messaging.subscribe(StartSeparateVideoMessageModel, booking_repeat_event)

        await rabbitmq_messaging.install(queue_name="video.separate")
