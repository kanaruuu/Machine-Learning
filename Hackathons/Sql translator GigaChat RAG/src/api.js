export const fetchHello = async () => {
    const response = await fetch('http://localhost:8000/');
    const data = await response.json();
    return data;
  };
  