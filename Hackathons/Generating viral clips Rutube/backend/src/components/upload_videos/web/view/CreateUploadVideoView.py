import os
import uuid

import aioboto3
from fastapi import HTTPException, UploadFile, File
from pathlib import Path
from starlette.responses import JSONResponse

from components.upload_videos.domain.models.StartSeparateVideoMessageModel import StartSeparateVideoMessageModel
from components.upload_videos.functions.core.IAddRequestUploadVideoFunction import IAddRequestUploadVideoFunction
from infrastructure.amqp_messaging.common.core.IRabbitFactory import IRabbitFactory
from infrastructure.amqp_messaging.common.core.IRabbitMessaging import IRabbitMessaging
from infrastructure.object_storage.s3_service.core.IS3ServiceFactory import IS3ServiceFactory


class CreateUploadVideoView:

    def __init__(
        self,
        s3_service_factory: IS3ServiceFactory,
        amqp_messaging_factory: IRabbitFactory,
        add_request_upload_video_function: IAddRequestUploadVideoFunction
    ):
        self.__s3_service_factory = s3_service_factory
        self.__amqp_messaging_factory = amqp_messaging_factory
        self.__add_request_upload_video_function = add_request_upload_video_function

    async def __call__(
        self,
        type_transform: str,
        file: UploadFile = File(...),
    ):
        s3_service = self.__s3_service_factory()
        amqp_messaging: IRabbitMessaging = await self.__amqp_messaging_factory()

        CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB
        MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024 # 5 GB

        filename = file.filename
        file_extension = os.path.splitext(filename)[1]

        allowed_extensions = [".mp4", ".avi", ".mp3", ".mov", ".mkv"]
        if file_extension.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла.")

        unique_filename = f"{uuid.uuid4()}{file_extension}"

        upload_id = None
        total_size = 0

        try:
            upload_id = await s3_service.create_multipart_upload(
                bucket_name="test-default-bucket",
                file_key=unique_filename,
                acl='public-read'
            )

            parts = []
            part_number = 1

            while True:

                data = await file.read(CHUNK_SIZE)
                if not data:
                    break

                total_size += len(data)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400,
                                        detail="Размер файла превышает максимально допустимый размер 1 GB.")

                part_response = await s3_service.upload_part(
                    bucket_name="test-default-bucket",
                    file_key=unique_filename,
                    part_number=part_number,
                    upload_id=upload_id,
                    data=data
                )

                parts.append({
                    'PartNumber': part_number,
                    'ETag': part_response['ETag']
                })

                part_number += 1

            if not parts:
                await s3_service.put_object(
                    bucket_name="test-default-bucket",
                    object_name=unique_filename,
                    data=b'',
                    acl='public-read'
                )
            else:
                await s3_service.complete_multipart_upload(
                    bucket_name="test-default-bucket",
                    file_key=unique_filename,
                    upload_id=upload_id,
                    multipart_upload={'Parts': parts}
                )

            message_id = uuid.uuid4()

            await amqp_messaging.publish(
                StartSeparateVideoMessageModel(
                    message_id=message_id,
                    file_key=unique_filename,
                    type_transform=type_transform
                ),
                routing_key="video.separate"
            )

            await self.__add_request_upload_video_function(
                request_id=str(message_id),
                filekey=unique_filename
            )

            # Возвращаем успешный ответ
            return {"filename": unique_filename, "request_id": str(message_id)}

        except Exception as e:
            # В случае ошибки, прерываем multipart upload
            if upload_id:
                await s3_service.abort_multipart_upload(
                    bucket_name="test-default-bucket",
                    file_key=unique_filename,
                    upload_id=upload_id
                )
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")
