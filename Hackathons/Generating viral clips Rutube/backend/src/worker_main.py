import asyncio
import logging
import os

import whisper
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from components.separator_video.factories.core.ISubtitlesModelFactory import ISubtitlesModelFactory
from components.separator_video.factories.impl.SubtitlesModelFactory import SubtitlesModelFactory
from components.separator_video.worker.impl.SeparateVideoWorker import SeparateVideoWorker
from infrastructure.amqp_messaging.common.impl.RabbitFactory import RabbitFactory
from infrastructure.amqp_messaging.models.RabbitMqSettings import RabbitMqSettings
from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from infrastructure.database_connection_pool.factories.impl.SessionFactory import SessionFactory
from infrastructure.object_storage.s3_service.core.IS3ServiceFactory import IS3ServiceFactory
from infrastructure.object_storage.s3_service.impl.S3ServiceFactory import S3ServiceFactory
from infrastructure.object_storage.s3_service.models.S3SettingsModel import S3SettingsModel
from system.ioc.ioc import ioc


async def startup():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    ioc.set(ISubtitlesModelFactory, SubtitlesModelFactory(
        model=whisper.load_model("medium", device="cuda")
    ))

    ioc.set(IS3ServiceFactory, S3ServiceFactory(
        s3_settings=S3SettingsModel(
            aws_access_key_id=os.getenv('S3_ACCESS'),
            aws_secret_access_key=os.getenv('S3_SECRET'),
            endpoint_url=os.getenv('S3_URL'),
        )
    ))

    ioc.set(ISessionFactory, SessionFactory(
        session_maker=sessionmaker(
            bind=create_async_engine(
                "postgresql+asyncpg://example:example@db:5432/example",
                echo=True,
            ),
            class_=AsyncSession,
            expire_on_commit=False
        )
    ))

    rabbit_factory = RabbitFactory(settings=RabbitMqSettings(
        rabbitmq_host=os.getenv('RABBITMQ_HOST'),
        rabbitmq_user=os.getenv('RABBITMQ_USER'),
        rabbitmq_pass=os.getenv('RABBITMQ_PASS'),
        rabbitmq_port=os.getenv('RABBITMQ_PORT'),
        rabbitmq_vhost=os.getenv('RABBITMQ_VHOST')
    ))

    separate_video_worker = SeparateVideoWorker(rabbitmq_messaging_factory=await rabbit_factory())

    logging.info("START LISTEN SEPARATE VIDEO WORKER")
    await separate_video_worker.start()



if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.get_event_loop().run_until_complete(
        startup()
    )
