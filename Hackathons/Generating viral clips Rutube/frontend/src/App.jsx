import "nouislider/distribute/nouislider.css";
import { VideoProvider } from "./hooks/videoProvider";
import Player from "./components/Player";

const App = () => {
  return (
    <VideoProvider>
      <Player />
    </VideoProvider>
  );
};

export default App;
