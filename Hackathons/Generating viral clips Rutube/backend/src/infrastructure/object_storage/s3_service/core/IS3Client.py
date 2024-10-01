from typing import Optional


class IS3Client:

    async def create_bucket(self, bucket_name: str):
        raise NotImplementedError()

    async def upload_file(self, bucket_name: str, object_name: str, file_path: str):
        raise NotImplementedError()

    async def upload_file_object(self, bucket_name: str, file_key: str, file_bytes: bytes):
        raise NotImplementedError()

    async def download_file(self, bucket_name: str, object_name: str, file_path: str):
        raise NotImplementedError()

    async def get_object(self, bucket_name: str, object_name: str) -> bytes:
        raise NotImplementedError()

    async def get_link_object(self, bucket_name: str, file_key: str):
        raise NotImplementedError()

    async def put_object(self, bucket_name: str, object_name: str, data: bytes, acl: str):
        raise NotImplementedError()

    async def create_multipart_upload(self, bucket_name: str, file_key: str, acl: str) -> int:
        raise NotImplementedError()

    async def upload_part(self, bucket_name: str, file_key: str, part_number: int, upload_id: int, data: bytes) -> dict:
        raise NotImplementedError()

    async def complete_multipart_upload(self, bucket_name: str, file_key: str, upload_id: int, multipart_upload: dict):
        raise NotImplementedError()

    async def abort_multipart_upload(self, bucket_name: str, file_key: str, upload_id: int):
        raise NotImplementedError()