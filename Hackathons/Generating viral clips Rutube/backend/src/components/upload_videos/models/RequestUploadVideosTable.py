from sqlalchemy import Column, Integer, String, UUID, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class RequestUploadVideosTable(Base):
    __tablename__ = 'request_upload_videos'

    id = Column(Integer, primary_key=True, index=True)

    request_id = Column(UUID)
    filekey = Column(String)
    is_complete = Column(Boolean)