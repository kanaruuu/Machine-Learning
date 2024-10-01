import io

import aioboto3

from ..core.IS3Client import IS3Client
from ..models.S3SettingsModel import S3SettingsModel


class S3Client(IS3Client):
    def __init__(
        self,
        s3_settings: S3SettingsModel
    ):
        self.__s3_settings = s3_settings
        self.session = aioboto3.Session(
            aws_access_key_id=s3_settings.aws_access_key_id,
            aws_secret_access_key=s3_settings.aws_secret_access_key,
        )

    async def create_bucket(self, bucket_name: str):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            await s3_client.create_bucket(
                Bucket=bucket_name
            )

    async def upload_file(self, bucket_name: str, object_name: str, file_path: str):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            await s3_client.upload_file(file_path, bucket_name, object_name)

    async def upload_file_object(self, bucket_name: str, file_key: str, file_bytes: bytes):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            await s3_client.upload_fileobj(io.BytesIO(file_bytes), bucket_name, file_key)

    async def download_file(self, bucket_name: str, object_name: str, file_path: str):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            await s3_client.download_file(bucket_name, object_name, file_path)

    async def get_object(self, bucket_name: str, object_name: str) -> bytes:
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            response = await s3_client.get_object(Bucket=bucket_name, Key=object_name)
            data = await response['Body'].read()
            return data

    async def get_link_object(self, bucket_name: str, file_key: str):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            url = await s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': file_key
                },
                ExpiresIn=None
            )
            return url

    async def put_object(self, bucket_name: str, object_name: str, data: bytes, acl: str):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            await s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=data, ACL=acl)

    async def create_multipart_upload(self, bucket_name: str, file_key: str, acl: str) -> int:
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            response = await s3_client.create_multipart_upload(
                Bucket=bucket_name,
                Key=file_key,
                ACL=acl
            )
            upload_id = response['UploadId']
        return upload_id

    async def upload_part(self, bucket_name: str, file_key: str, part_number: int, upload_id: int, data: bytes) -> dict:
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            part_response = await s3_client.upload_part(
                Bucket=bucket_name,
                Key=file_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=data
            )
        return part_response

    async def complete_multipart_upload(self, bucket_name: str, file_key: str, upload_id: int, multipart_upload: dict):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            await s3_client.complete_multipart_upload(
                Bucket=bucket_name,
                Key=file_key,
                UploadId=upload_id,
                MultipartUpload=multipart_upload
            )

    async def abort_multipart_upload(self, bucket_name: str, file_key: str, upload_id: int):
        async with self.session.client('s3', endpoint_url=self.__s3_settings.endpoint_url) as s3_client:
            await s3_client.abort_multipart_upload(
                Bucket=bucket_name,
                Key=file_key,
                UploadId=upload_id
            )