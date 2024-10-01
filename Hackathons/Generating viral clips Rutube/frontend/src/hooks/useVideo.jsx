import { useContext } from "react";
import { VideoContext } from "./videoProvider";

const useVideo = () => {
  const videoContext = useContext(VideoContext);

  if (videoContext === undefined) {
    throw new Error("useVideo must be used within a VideoProvider");
  }

  return videoContext;
};

export default useVideo;
