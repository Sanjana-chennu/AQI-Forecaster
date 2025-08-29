// src/App.js

import React, { useState, useEffect, useRef } from 'react';
import './App.css'; // We will create this for styling

function App() {
  const [messages, setMessages] = useState([
    { text: "Hello! How can I help you with the air quality forecast today?", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const chatBoxEndRef = useRef(null);

  // Effect to automatically scroll to the bottom when new messages are added
  useEffect(() => {
    chatBoxEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSendMessage = async () => {
    const userMessage = input.trim();
    if (!userMessage || isLoading) return;

    // 1. Add user message to the chat and clear the input
    setMessages(prev => [...prev, { text: userMessage, sender: "user" }]);
    setInput("");
    setIsLoading(true);

    try {
      // 2. Send the message to our FastAPI backend API
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // 3. Add the bot's response to the chat
      setMessages(prev => [...prev, { text: data.reply, sender: "bot" }]);

    } catch (error) {
      console.error("Fetch error:", error);
      setMessages(prev => [...prev, { text: "Sorry, I couldn't connect to the server. Please ensure the backend is running.", sender: "bot" }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`message-wrapper ${msg.sender}-wrapper`}>
            <div className={`message ${msg.sender}-message`}>
              {msg.text}
            </div>
          </div>
        ))}
        {isLoading && (
            <div className="message-wrapper bot-wrapper">
                <div className="message bot-message thinking">
                    <span>.</span><span>.</span><span>.</span>
                </div>
            </div>
        )}
        <div ref={chatBoxEndRef} />
      </div>
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Ask for a forecast..."
          disabled={isLoading}
        />
        <button onClick={handleSendMessage} disabled={isLoading}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;