from infrastructure.amqp_messaging.models.BaseModelMessage import BaseModelMessage


class StartSeparateVideoMessageModel(BaseModelMessage):
    file_key: str
    type_transform: str