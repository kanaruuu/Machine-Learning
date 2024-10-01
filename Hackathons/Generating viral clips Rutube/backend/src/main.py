import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from components.upload_videos.web.UploadVideoWebInstall import UploadVideoWebInstall
from infrastructure.amqp_messaging.common.core.IRabbitFactory import IRabbitFactory
from infrastructure.amqp_messaging.common.impl.RabbitFactory import RabbitFactory
from infrastructure.amqp_messaging.models.RabbitMqSettings import RabbitMqSettings
from infrastructure.database_connection_pool.factories.core.ISessionFactory import ISessionFactory
from infrastructure.database_connection_pool.factories.impl.SessionFactory import SessionFactory
from infrastructure.object_storage.s3_service.core.IS3ServiceFactory import IS3ServiceFactory
from infrastructure.object_storage.s3_service.impl.S3ServiceFactory import S3ServiceFactory
from infrastructure.object_storage.s3_service.models.S3SettingsModel import S3SettingsModel
from system.ioc.ioc import ioc


@asynccontextmanager
async def lifespan(app: FastAPI):

    ioc.set(IS3ServiceFactory, S3ServiceFactory(
        s3_settings=S3SettingsModel(
            aws_access_key_id=os.getenv('S3_ACCESS'),
            aws_secret_access_key=os.getenv('S3_SECRET'),
            endpoint_url=os.getenv('S3_URL'),
        )
    ))

    rabbit_factory = RabbitFactory(settings=RabbitMqSettings(
        rabbitmq_host=os.getenv('RABBITMQ_HOST'),
        rabbitmq_user=os.getenv('RABBITMQ_USER'),
        rabbitmq_pass=os.getenv('RABBITMQ_PASS'),
        rabbitmq_port=os.getenv('RABBITMQ_PORT'),
        rabbitmq_vhost=os.getenv('RABBITMQ_VHOST')
    ))
    ioc.set(IRabbitFactory, await rabbit_factory())

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

    assist_web_install = UploadVideoWebInstall()

    assist_web_install(app=app)
    yield

app = FastAPI(
    lifespan=lifespan,
    title = "Assistant API",
    version = "0.1.0"
)

app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
