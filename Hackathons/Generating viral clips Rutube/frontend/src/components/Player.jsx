import {
  DownloadOutlined,
  PauseOutlined,
  PlayCircleOutlined,
  DeleteOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { Button, Spin, Typography, Divider, Modal, List, Space } from "antd";
import Nouislider from "nouislider-react";
import "nouislider/distribute/nouislider.css";
import { useEffect, useRef, useState } from "react";
import useVideo from "../hooks/useVideo";
import DropFile from "./DropFile";
import axios from "axios";
import ListItem from "./ListItem";

const { Text } = Typography;

let ffmpeg;

const Player = () => {
  const [endTime, setEndTime] = useState(0);
  const [startTime, setStartTime] = useState(0);

  const { videoSrc, setVideoSrc, url, fileData, standartVideoSrc } = useVideo();

  const [isScriptLoading, setIsScriptLoading] = useState(false);

  const [videoDuration, setVideoDuration] = useState(0);

  const videoRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);

  const [isModalVisible, setIsModalVisible] = useState(false);

  const [listData, setListData] = useState(null);

  const [listLoading, setListLoading] = useState(false);

  const [listVideoData, setListVideoData] = useState(null);

  const [selectedListItem, setSelectedListItem] = useState(null);

  let initialSliderValue = 0;

  const loadScript = (src) => {
    return new Promise((onFulfilled, _) => {
      const script = document.createElement("script");
      let loaded;
      script.async = "async";
      script.defer = "defer";
      script.setAttribute("src", src);
      script.onreadystatechange = script.onload = () => {
        if (!loaded) {
          onFulfilled();
        }
        loaded = true;
      };
      script.onerror = function () {
        console.log("Script failed to load");
      };
      document.getElementsByTagName("head")[0].appendChild(script);
    });
  };

  useEffect(() => {
    loadScript(
      "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.11.2/dist/ffmpeg.min.js"
    ).then(() => {
      if (typeof window !== "undefined") {
        ffmpeg = window.FFmpeg.createFFmpeg({
          log: true,
        });
        ffmpeg.load();
        setIsScriptLoading(true);
      }
    });
  }, []);

  useEffect(() => {
    const currentVideo = videoRef.current;

    const handleLoadedMetadata = () => {
      console.log(
        "Metadata loaded:",
        currentVideo.videoWidth,
        currentVideo.videoHeight
      );
      setVideoDuration(currentVideo.duration);
      setEndTime(currentVideo.duration);
    };

    const handleError = () => {
      console.error("Error loading video:", currentVideo.error);
    };

    if (currentVideo) {
      currentVideo.addEventListener("loadedmetadata", handleLoadedMetadata);
      currentVideo.addEventListener("error", handleError);
    }

    return () => {
      if (currentVideo) {
        currentVideo.removeEventListener(
          "loadedmetadata",
          handleLoadedMetadata
        );
        currentVideo.removeEventListener("error", handleError);
      }
    };
  }, [videoSrc]);

  const handlePauseVideo = () => {};
  const updateOnSliderChange = (values, handle) => {
    // setVideoTrimmedUrl("");
    let readValue;

    if (handle) {
      readValue = values[handle] | 0;
      if (endTime !== readValue) {
        setEndTime(readValue);
      }
    } else {
      readValue = values[handle] | 0;
      if (initialSliderValue !== readValue) {
        initialSliderValue = readValue;
        if (videoRef && videoRef.current) {
          videoRef.current.currentTime = readValue;
          setStartTime(readValue);
        }
      }
    }
    handlePlay();
  };

  const convertToHHMMSS = (val) => {
    const secNum = parseInt(val, 10);
    let hours = Math.floor(secNum / 3600);
    let minutes = Math.floor((secNum - hours * 3600) / 60);
    let seconds = secNum - hours * 3600 - minutes * 60;

    if (hours < 10) {
      hours = "0" + hours;
    }
    if (minutes < 10) {
      minutes = "0" + minutes;
    }
    if (seconds < 10) {
      seconds = "0" + seconds;
    }
    let time;

    time = hours + ":" + minutes + ":" + seconds;
    return time;
  };

  const handlePlay = () => {
    if (videoRef && videoRef.current) {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handlePause = () => {
    if (videoRef && videoRef.current) {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleSave = async () => {
    // Останавливаем видео перед обработкой
    if (videoRef && videoRef.current) {
      videoRef.current.pause();
      setIsPlaying(false);
    }

    // Проверяем, был ли загружен ffmpeg
    if (!ffmpeg.isLoaded()) {
      await ffmpeg.load();
    }

    // Загружаем исходное видео в память FFmpeg
    const videoUrl = videoSrc;
    const response = await fetch(videoUrl);
    const videoData = await response.arrayBuffer();

    ffmpeg.FS("writeFile", "input.mp4", new Uint8Array(videoData));

    // Выполняем команду обрезки видео
    await ffmpeg.run(
      "-i",
      "input.mp4", // Входное видео
      "-ss",
      convertToHHMMSS(startTime), // Начало обрезки
      "-to",
      convertToHHMMSS(endTime), // Конец обрезки
      "-c",
      "copy", // Кодек, используемый для копирования видео
      "output.mp4" // Выходной файл
    );

    // Извлекаем обрезанный файл
    const trimmedVideoData = ffmpeg.FS("readFile", "output.mp4");

    // Создаем Blob для скачивания
    const trimmedBlob = new Blob([trimmedVideoData.buffer], {
      type: "video/mp4",
    });

    // Генерируем ссылку для скачивания
    const trimmedVideoUrl = URL.createObjectURL(trimmedBlob);

    // Создаем ссылку для скачивания файла
    const a = document.createElement("a");
    a.href = trimmedVideoUrl;
    a.download = "trimmed-video.mp4";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    // Освобождаем память
    URL.revokeObjectURL(trimmedVideoUrl);
  };

  const handleDelete = () => {
    if (videoRef && videoRef.current) {
      videoRef.current.pause(); // Остановить видео
    }
    setIsPlaying(false); // Сбросить состояние воспроизведения
    setVideoSrc(null); // Очистить источник видео
    setStartTime(0); // Сбросить время начала
    setEndTime(0); // Сбросить время конца
    setVideoDuration(0); // Сбросить продолжительность видео
    initialSliderValue = 0; // Сбросить ползунок
  };

  useEffect(() => {
    if (videoSrc) {
      const fetchData = async () => {
        setListLoading(true);
        try {
          // Одновременное выполнение обоих запросов
          const [statusResponse, processedResponse] = await Promise.all([
            axios.get(url + "/video/status?request_id=" + fileData.request_id),
            axios.get(
              url + "/video/processed/list?request_id=" + fileData.request_id
            ),
          ]);

          // Проверка статуса
          if (statusResponse.data.is_complete) {
            setListLoading(false);
            setListData(statusResponse.data);
            setListVideoData(processedResponse.data);
            // Остановка запросов
            clearInterval(intervalId);
          } else {
            // Если статус не true, обновляем состояние видео
            setListVideoData(processedResponse.data);
          }
        } catch (error) {
          console.log(error);
          setListLoading(false); // Устанавливаем состояние загрузки в false в случае ошибки
        }
      };

      const intervalId = setInterval(fetchData, 2000);
      fetchData(); // Сразу вызываем функцию, чтобы не ждать 2 секунды на первый запрос

      return () => clearInterval(intervalId); // Очистка интервала при размонтировании
    }
  }, [fileData?.request_id, url, videoSrc]);

  return isScriptLoading ? (
    <div>
      <div
        style={{
          display: "flex",
          width: "100%",
          maxWidth: 1200,
          margin: "0 auto",
          marginTop: 40,
        }}
      >
        <div
          style={{
            width: 450,
            height: 530,
            backgroundColor: "#e4e4e4",
            borderRadius: 16,
            display: "flex",
            marginRight: 30,
            flexDirection: "column",
          }}
        >
          <Divider style={{ height: 20 }}>Ролики</Divider>
          {standartVideoSrc && (
            <ListItem
              title={"Стандартное видео"}
              onClick={() => {
                setSelectedListItem(standartVideoSrc);
                setVideoSrc(null);
              }}
            />
          )}

          {listLoading ? (
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                height: 510,
              }}
            >
              <Spin size="large" />
            </div>
          ) : (
            listData &&
            listVideoData && (
              <List
                dataSource={listVideoData}
                renderItem={(item, key) => (
                  <ListItem
                    onClick={() => {
                      setSelectedListItem(item);
                      setVideoSrc(item.video_url);
                      console.log(item);
                      setTimeout(() => {
                        handlePlay(); // Начинаем воспроизведение видео после установки источника
                      }, 100);
                    }}
                    title={`Элемент ${key + 1}`}
                  />
                )}
              />
            )
          )}
        </div>
        {videoSrc ? (
          <div
            style={{
              width: 300,
              height: 530,
              backgroundColor: !videoSrc ? "#e4e4e4" : "black",
              borderRadius: 16,
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              marginRight: 30,
            }}
          >
            <video
              ref={videoRef}
              src={videoSrc}
              onTimeUpdate={handlePauseVideo}
              style={{ borderRadius: 16, width: "100%", height: "100%" }}
              crossOrigin="anonymous"
            />
          </div>
        ) : (
          <Button
            style={{
              marginRight: 30,
              width: 300,
              height: 530,
              backgroundColor: "#e4e4e4",
            }}
            icon={<UploadOutlined />}
            onClick={() => setIsModalVisible(true)}
          >
            Нажмите для загрузки видео
          </Button>
        )}

        <div style={{ width: 450 }}>
          <div
            style={{
              width: "100%",
              height: 400,
              backgroundColor: "#e4e4e4",
              borderRadius: 16,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Divider style={{ height: 20 }}>Информация</Divider>
            {selectedListItem && (
              <div>
                <Space>
                  <Typography.Text style={{ marginRight: 10 }}>
                    metrics:
                  </Typography.Text>
                  <Typography.Text strong>
                    {selectedListItem.metrics}
                  </Typography.Text>
                </Space>
                <Space>
                  <Typography.Text style={{ marginRight: 10 }}>
                    {selectedListItem.metrics_fields}
                  </Typography.Text>
                </Space>
              </div>
            )}
          </div>
          <Button
            icon={<DownloadOutlined />}
            size="large"
            onClick={handleSave}
            style={{
              width: "100%",
              marginTop: 20,
              height: 40,
              backgroundColor: "#e4e4e4",
            }}
            disabled={!videoSrc}
          >
            Скачать
          </Button>
          <Button
            danger
            disabled={!videoSrc}
            icon={<DeleteOutlined />}
            onClick={handleDelete}
            style={{
              width: "100%",
              marginTop: 20,
              height: 40,
              backgroundColor: "#e4e4e4",
            }}
          >
            Удалить
          </Button>
        </div>
      </div>

      {videoSrc && (
        <div
          style={{
            position: "fixed",
            bottom: 0,
            width: "100%",
            backgroundColor: "#e4e4e4",
            paddingTop: 10,
            border: "1px solid #bebdbd",
          }}
        >
          <div
            style={{
              maxWidth: "100%",
              borderBottom: "1px solid #bebdbd",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                maxWidth: 1200,
                margin: "0 auto",
                paddingBottom: 10,
              }}
            >
              <div>
                {!isPlaying ? (
                  <Button
                    style={{ marginRight: 10 }}
                    icon={<PlayCircleOutlined />}
                    onClick={handlePlay}
                    size="large"
                  />
                ) : (
                  <Button
                    style={{ marginRight: 10 }}
                    icon={<PauseOutlined />}
                    onClick={handlePause}
                    size="large"
                  />
                )}

                <Text style={{ marginRight: 5 }} strong>
                  {convertToHHMMSS(startTime)}
                </Text>
                <Text disabled style={{ marginRight: 5 }}>
                  |
                </Text>
                <Text disabled>{convertToHHMMSS(endTime)}</Text>
              </div>
            </div>
          </div>
          <div
            style={{
              maxWidth: 1200,
              margin: "0 auto",
              paddingTop: 20,
              paddingBottom: 20,
            }}
          >
            <Nouislider
              behaviour="tap-drag"
              step={1}
              range={{ min: 0, max: videoDuration || 2 }}
              start={[0, videoDuration || 2]}
              connect
              onUpdate={updateOnSliderChange}
            />
          </div>

          {/* <button onClick={handleAddSection}>Add section</button> */}
        </div>
      )}
      <Modal
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={null}
        centered
        title={"Загрузка видео"}
      >
        <DropFile />
      </Modal>
    </div>
  ) : (
    <Spin />
  );
};

export default Player;
