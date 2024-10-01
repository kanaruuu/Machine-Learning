import { InboxOutlined } from "@ant-design/icons";
import { Upload, Radio, Space, Typography } from "antd";
import axios from "axios";
import useVideo from "../hooks/useVideo";
import { useState } from "react";

const { Dragger } = Upload;

const DropFile = () => {
  const { handleFileUpload, url, setFileData } = useVideo();
  const [transform, setTransform] = useState(null);
  const customRequest = async ({ file, onSuccess, onError }) => {
    const formData = new FormData();
    formData.append("file", file);

    axios
      .post(url + "/video/upload?type_transform=" + transform, formData)
      .then((response) => {
        console.log("Response received:", response);
        if (response.status === 201) {
          const data = response.data;
          setFileData(data);
          console.log(data);
          onSuccess(data);
        } else {
          throw new Error(
            "Ошибка при загрузке файла. Статус: " + response.status
          );
        }
      })
      .catch((error) => {
        console.error("Error occurred:", error);
        onError(error);
      })
      .finally(() => {
        setTransform(null);
      });
  };

  const props = {
    name: "file",
    multiple: true,
    customRequest,
    accept: "video/mp4",
    progress: true,
    onChange(info) {
      // Call the onChange prop with the file
      handleFileUpload(info.file);
    },
    onDrop(e) {
      console.log("Dropped files", e.dataTransfer.files);
    },
    disabled: transform === null,
    style: {
      width: "100%",
    },
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        paddingBottom: 30,
      }}
    >
      <Typography.Title style={{ margin: 0, marginBottom: 20 }} level={3}>
        Выберите тип преобразования
      </Typography.Title>
      <Radio.Group
        value={transform}
        onChange={(e) => setTransform(e.target.value)}
        style={{ marginBottom: 20 }}
      >
        <Space direction="vertical">
          <Radio value={"quick"}>
            Создать вертикальное видео с фокусом на говорящем без субтитров:
            Быстро и эффективно
          </Radio>
          <Radio value={"slow"}>
            Создать видео с субтитрами и черными полосами по вертикали:
            Тщательно и профессионально
          </Radio>
        </Space>
      </Radio.Group>
      <Dragger {...props}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          Нажмите или перетащите видео для загрузки
        </p>
      </Dragger>
    </div>
  );
};

export default DropFile;
