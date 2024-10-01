from sqlalchemy import Column, Integer, String, UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class FinishedProcessedRequestTable(Base):
    __tablename__ = 'finished_processed_request'

    id = Column(Integer, primary_key=True, index=True)

    request_id = Column(UUID)
    filekey = Column(String)
    metrics = Column(String)
    metrics_fields = Column(String)