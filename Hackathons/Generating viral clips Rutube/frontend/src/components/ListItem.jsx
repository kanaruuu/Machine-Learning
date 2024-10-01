import { Button } from "antd";

// eslint-disable-next-line react/prop-types
const ListItem = ({ title, onClick }) => {
  // eslint-disable-next-line react/prop-types
  return (
    <Button onClick={onClick} style={{ margin: 5, height: 50 }}>
      {title}
    </Button>
  );
};

export default ListItem;
