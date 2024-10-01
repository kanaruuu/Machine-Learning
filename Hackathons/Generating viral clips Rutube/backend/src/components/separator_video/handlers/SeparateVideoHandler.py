import asyncio
from math import ceil
from typing import Mapping, Any, Optional

from aio_pika import IncomingMessage

import xml.etree.ElementTree as ET

from components.separator_video.domain.functions.delete_useless_scenes import filter_scenes, \
    get_video_duration_from_bytes
from components.separator_video.domain.functions.describe_model import generate_fields
from components.separator_video.domain.functions.generate_clips_from_scenes import run_process
from components.separator_video.domain.functions.generate_video_from_shots import get_video_scenes_bytes
from components.separator_video.domain.functions.metrics import start_proccess
from components.separator_video.domain.functions.shot_to_scenes import process_video
from components.separator_video.domain.functions.simple_subtitles import transcribe_video_bytes
from components.separator_video.domain.functions.speaker_detection import speaker_detect
from components.separator_video.domain.functions.split_and_save_videos import merge_shots_into_scenes
from components.separator_video.domain.functions.subtitles import make_subtitled_video
from components.separator_video.domain.functions.top_dialogues import get_top_dialogues_tfidf
from components.separator_video.factories.core.ISubtitlesModelFactory import ISubtitlesModelFactory
from components.separator_video.functions.core.IAddProcessedVideosFunction import IAddProcessedVideosFunction
from components.separator_video.functions.core.IChangeStatusRequestFunction import IChangeStatusRequestFunction
from components.upload_videos.domain.models.StartSeparateVideoMessageModel import StartSeparateVideoMessageModel
from infrastructure.amqp_messaging.common.core.EventHandler import IEventHandler
from infrastructure.object_storage.s3_service.core.IS3ServiceFactory import IS3ServiceFactory


class SeparateVideoHandler(IEventHandler[StartSeparateVideoMessageModel]):

    def __init__(
        self,
        subtitles_model_factory: ISubtitlesModelFactory,
        s3_service_factory: IS3ServiceFactory,
        change_status_request_function: IChangeStatusRequestFunction,
        add_processed_videos_function: IAddProcessedVideosFunction
    ):
        self.__change_status_request_function = change_status_request_function
        self.__subtitles_model_factory = subtitles_model_factory
        self.__s3_service_factory = s3_service_factory
        self.__add_processed_videos_function = add_processed_videos_function

    async def __call__(self, event: Mapping[str, Any], message: IncomingMessage):
        video_message = StartSeparateVideoMessageModel(**event)

        s3_service = self.__s3_service_factory()

        video_bytes = await s3_service.get_object(
            bucket_name="test-default-bucket",
            object_name=video_message.file_key
        )

        loop = asyncio.get_running_loop()

        return_list, temp_input_filepath = await loop.run_in_executor(None, process_video, video_bytes)
        shots_list = await loop.run_in_executor(None, get_video_scenes_bytes, return_list, temp_input_filepath)
        scenes_list = await loop.run_in_executor(None, merge_shots_into_scenes, shots_list)

        filtered_scenes = await loop.run_in_executor(None, filter_scenes, scenes_list)

        for index, scene in enumerate(filtered_scenes):
            print(f"WORK WITH {index} VIDEO")
            video_bytes = await loop.run_in_executor(None, speaker_detect, scene)

            subtitles_text = await loop.run_in_executor(
                None, transcribe_video_bytes, video_bytes, self.__subtitles_model_factory()
            ) # Text

            video_bytes_with_subtitles = await loop.run_in_executor(
                None, make_subtitled_video, video_bytes, self.__subtitles_model_factory()
            )

            metrics = await loop.run_in_executor(
                None, start_proccess, subtitles_text
            )

            metrics_fields = await loop.run_in_executor(
                None, generate_fields, subtitles_text
            )

            await s3_service.put_object(
                bucket_name="test-default-bucket",
                object_name=f"{index}_{video_message.file_key}",
                data=video_bytes_with_subtitles,
                acl='public-read'
            )

            await self.__add_processed_videos_function(
                request_id=str(video_message.message_id),
                filekey=f"{index}_{video_message.file_key}",
                metrics=str(metrics),
                metrics_fields=str(metrics_fields)
            )

        await self.__change_status_request_function(
            request_id=str(video_message.message_id)
        )
