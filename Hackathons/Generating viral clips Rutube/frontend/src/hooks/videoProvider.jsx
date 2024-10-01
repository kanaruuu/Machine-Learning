import { createContext, useState } from "react";

export const VideoContext = createContext();

// eslint-disable-next-line react/prop-types
export const VideoProvider = ({ children }) => {
  const [videoSrc, setVideoSrc] = useState(null);
  const [standartVideoSrc, setStandartVideoSrc] = useState(null);
  const [videoFileValue, setVideoFileValue] = useState(null);
  const url = "http://109.71.15.18:8005";
  const [fileData, setFileData] = useState(null);

  const handleFileUpload = (e) => {
    let file = e.originFileObj;
    const blobURL = URL.createObjectURL(file);
    setVideoSrc(blobURL);
    setStandartVideoSrc(blobURL);
    setVideoFileValue(file);
  };

  return (
    <VideoContext.Provider
      value={{
        videoSrc,
        setVideoSrc,
        videoFileValue,
        setVideoFileValue,
        handleFileUpload,
        url,
        fileData,
        setFileData,
        standartVideoSrc,
      }}
    >
      {children}
    </VideoContext.Provider>
  );
};
