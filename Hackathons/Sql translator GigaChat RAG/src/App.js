import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { solarizedlight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { marked } from 'marked';

const LoadingIndicator = () => (
  <div className="loading-indicator">
    <div className="loader"></div>
    <span>Загрузка...</span>
  </div>
);

const MessageContent = ({ text, isUser, onCheckClicked, isLoading, isLastMessage, hasSQLCode, hasBeenChecked }) => {
  const sqlParts = text.match(/```sql[\s\S]*?```/g);
  const parts = text.split(/```sql[\s\S]*?```/g).filter(part => part);
  const sqlCode = sqlParts ? sqlParts.map(part => part.replace(/```sql\n|```/g, '')) : [];

  const handleCopy = (code) => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className={`MessageBubble ${isUser ? 'user' : ''}`}>
      {isLoading && isLastMessage ? (
        <LoadingIndicator />
      ) : (
        <>
          {parts.map((part, index) => (
            <span key={index}>{part}</span>
          ))}
          {sqlCode.map((code, index) => (
            <div key={index} style={{ position: 'relative' }}>
              <SyntaxHighlighter 
                language="sql" 
                style={solarizedlight} 
                customStyle={{ 
                  width: 'auto',
                  height: 'auto',
                  padding: '10px',
                  borderRadius: '5px', 
                  fontSize: '14px',
                  margin: '10px 0',
                  display: 'block'
                }}
              >
                {code}
              </SyntaxHighlighter>
              <div className="button-container">
                <button onClick={() => handleCopy(code)} className="action-button">Копировать</button>
                <button 
                  onClick={onCheckClicked} 
                  className="action-button" 
                  disabled={isLoading || !hasSQLCode || hasBeenChecked}>
                  Проверить
                </button>
              </div>
            </div>
          ))}
        </>
      )}
      {isLoading && !isLastMessage && <div className="loading-animation">...</div>}
    </div>
  );
};

const WelcomeMessage = () => (
  <div className="WelcomeMessage">
    <h1>Добро пожаловать в SQL Консультанта!</h1>
    <p>Задайте свои вопросы по SQL-коду или отправьте запрос на проверку.</p>
  </div>
);

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [inputHeight, setInputHeight] = useState('auto');
  const [isLoading, setIsLoading] = useState(false);
  const [isChecking, setIsChecking] = useState(false);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState('GigaChat-Pro');
  const [showModelMenu, setShowModelMenu] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleInputChange = (event) => {
    const { value } = event.target;
    setInput(value);
    
    const lineCount = value.split('\n').length;
    const maxLines = isSidebarOpen ? 2 : 5;
    setInputHeight(`${Math.min(lineCount, maxLines) * 1.2}em`);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];

    if (!file || !file.name.endsWith('.sql')) {
      alert('Пожалуйста, загрузите файл с расширением .sql');
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      const fileContent = e.target.result;

      setMessages((prev) => [...prev, { text: `Отправка файла: ${file.name}`, isUser: true }]);
      
      setIsLoading(true);

      try {
        const response = await axios.post('http://localhost:8000/message/', { text: fileContent, model: selectedModel });
        setMessages((prev) => [...prev, { text: response.data.response, isUser: false }]);
      } catch (error) {
        console.error('Ошибка при отправке файла:', error);
      } finally {
        setIsLoading(false);
      }
    };
    reader.readAsText(file);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (input.trim() === '' || isLoading) return;

    const messageToSend = {
      text: input,
      model: selectedModel
    };

    setMessages((prev) => [...prev, { text: input, isUser: true }]);
    setInput('');
    setInputHeight('auto');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/message/', messageToSend);
      setMessages((prev) => [...prev, { text: response.data.response, isUser: false, hasBeenChecked: false }]);
    } catch (error) {
      console.error('Ошибка при отправке сообщения:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCheck = async (text, index) => {
    if (isChecking || messages[index].hasBeenChecked) return;

    setIsChecking(true);
    const loadingIndex = messages.length;  
    setMessages((prev) =>
      [...prev, { text: "Проверка...", isUser: false, isLoading: true }] 
    );

    const sqlParts = text.match(/```sql[\s\S]*?```/g);
    const sqlCodeToCheck = sqlParts ? sqlParts.map(part => part.replace(/```sql\n|```/g, '')).join('\n') : '';

    try {
      const response = await axios.post('http://localhost:8000/check_sql/', { text: sqlCodeToCheck });
      const resultMessage = response.data.result;  

      setMessages((prev) => {
        const updatedMessages = [...prev];
        updatedMessages[loadingIndex] = {
          text: resultMessage,
          isUser: false,
          isLoading: false,
          hasBeenChecked: true,
        };
        return updatedMessages;
      });
    } catch (error) {
      console.error('Ошибка при проверке SQL-кода:', error);
      setMessages((prev) => {
        const updatedMessages = [...prev];
        updatedMessages[loadingIndex] = {
          text: "Ошибка при проверке SQL-кода. Попробуйте снова.",
          isUser: false,
          isLoading: false,
          hasBeenChecked: true,
        };
        return updatedMessages;
      });
    } finally {
      setIsChecking(false);
    }
  };

  const handleModelClick = (model) => {
    setSelectedModel(model);
    setShowModelMenu(false);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <Router>
      <div className="AppContainer">
        <div className={`Sidebar ${isSidebarOpen ? '' : 'collapsed'}`}>
          <button className="ToggleButton" onClick={() => setSidebarOpen(!isSidebarOpen)}>
            {isSidebarOpen ? '←' : '→'}
          </button>
          {isSidebarOpen && (
            <div className="nav-buttons">
              <div>
                <button className="nav-button" onClick={() => setShowModelMenu(!showModelMenu)}>
                  Режим модели: {selectedModel}
                </button>
                {showModelMenu && (
                  <div className="model-dropdown">
                    <button className="model-option" onClick={() => handleModelClick('GigaChat-Pro')}>GigaChat-Pro</button>
                    <button className="model-option" onClick={() => handleModelClick('GigaChat-Plus')}>GigaChat-Plus</button>
                    <button className="model-option" onClick={() => handleModelClick('GigaChat')}>GigaChat</button>
                  </div>
                )}
              </div>
              <Link to="/" className="nav-button" style={{ textDecoration: 'none', color: 'inherit' }}>
                Чат
              </Link>
            </div>
          )}
        </div>
        <div className={`MessagesContainer ${isSidebarOpen ? 'shifted' : ''}`}>
          <Routes>
            <Route path="/" element={
              <>
                {messages.length === 0 && <WelcomeMessage />}
                {messages.map((msg, index) => {
                  const hasSQLCode = msg.text.includes('```sql') && msg.text.includes('```');
                  const hasBeenChecked = msg.hasBeenChecked || false;

                  return (
                    <MessageContent 
                      key={index} 
                      text={msg.text} 
                      isUser={msg.isUser}
                      onCheckClicked={() => handleCheck(msg.text, index)} 
                      isLoading={msg.isLoading} 
                      isLastMessage={index === messages.length - 1} 
                      hasSQLCode={hasSQLCode} 
                      hasBeenChecked={hasBeenChecked} 
                    />
                  );
                })}
                <div ref={messagesEndRef} />
              </>
            } />
          </Routes>
        </div>
        <form onSubmit={handleSubmit} className={`InputForm ${isSidebarOpen ? '' : 'shifted'}`}>
          <div className="InputContainer">
            <textarea
              className="Input"
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="О чём поговорим?"
              style={{ height: inputHeight }} 
              rows={1} 
            />
            <label htmlFor="file-upload" className="file-upload">
              <input 
                type="file" 
                id="file-upload" 
                className="file-input" 
                onChange={handleFileUpload} 
                style={{ display: 'none' }} 
              />
              <span className="plus-icon">+</span>
            </label>
            <button type="submit" className="Button" disabled={isLoading}>
              {isLoading ? 'Отправка...' : 'Отправить'}
            </button>
          </div>
        </form>
      </div>
    </Router>
  );
}

export default App;
