import React, { useState, useEffect, useRef } from "react";

const AMBER = "#a8681e";
const AMBER_DIM = "#c49050";

const formatMessage = (text) => {
  if (!text) return null;
  const parts = text.split(/(```[\s\S]*?```|\*\*[\s\S]*?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("```") && part.endsWith("```")) {
      const inner = part.slice(3, -3);
      const langMatch = inner.match(/^[\w-]+\n/);
      const lang = langMatch ? langMatch[0].trim() : null;
      const code = langMatch ? inner.slice(langMatch[0].length) : inner;
      return (
        <div
          key={i}
          style={{
            margin: "14px 0",
            borderRadius: "4px",
            overflow: "hidden",
            border: "1px solid #ddd7cf",
            background: "#eae5de",
          }}
        >
          {lang && (
            <div
              style={{
                padding: "6px 16px",
                fontSize: "10px",
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: AMBER_DIM,
                borderBottom: "1px solid #ddd7cf",
                fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              {lang}
            </div>
          )}
          <pre
            style={{
              padding: "16px",
              overflowX: "auto",
              fontSize: "12.5px",
              lineHeight: "1.75",
              color: "#4a5568",
              fontFamily: "'JetBrains Mono', monospace",
              margin: 0,
            }}
          >
            <code>{code}</code>
          </pre>
        </div>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} style={{ fontWeight: 500, color: "#1c1a17" }}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <span key={i}>{part}</span>;
  });
};

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [ws, setWs] = useState(null);
  const [wsStatus, setWsStatus] = useState("disconnected");
  const [isGenerating, setIsGenerating] = useState(false);
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws/chat");
  const [showConfig, setShowConfig] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const [sessionId] = useState(
    () => "session_" + Math.random().toString(36).substring(2, 9),
  );

  useEffect(() => {
    let socket;
    try {
      socket = new WebSocket(wsUrl);
      socket.onopen = () => setWsStatus("connected");
      socket.onclose = () => setWsStatus("disconnected");
      socket.onerror = () => setWsStatus("error");
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "start") {
          setIsGenerating(true);
          setMessages((prev) => [
            ...prev,
            { id: Date.now(), role: "ai", content: "" },
          ]);
        } else if (data.type === "chunk") {
          setMessages((prev) => {
            const msgs = [...prev];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "ai")
              msgs[msgs.length - 1] = {
                ...last,
                content: last.content + data.content,
              };
            return msgs;
          });
        } else if (data.type === "end") {
          setIsGenerating(false);
        } else if (data.type === "error") {
          setMessages((prev) => [
            ...prev,
            { id: Date.now(), role: "error", content: data.message },
          ]);
          setIsGenerating(false);
        }
      };
      setWs(socket);
    } catch {
      setWsStatus("error");
    }
    return () => {
      if (socket) socket.close();
    };
  }, [wsUrl]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || !ws || wsStatus !== "connected") return;
    ws.send(JSON.stringify({ message: input, session_id: sessionId }));
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: "user", content: input },
    ]);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const statusColor =
    wsStatus === "connected"
      ? "#5dba7a"
      : wsStatus === "error"
        ? "#c95050"
        : AMBER;
  const statusLabel =
    wsStatus === "connected"
      ? "Online"
      : wsStatus === "error"
        ? "Error"
        : "Connecting";

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=JetBrains+Mono:wght@400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body, #root { height: 100%; }
        ::-webkit-scrollbar { width: 2px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #d0c9c0; border-radius: 1px; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes statusPulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }
        @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }
        .msg-anim { animation: fadeUp 0.3s cubic-bezier(0.16,1,0.3,1) forwards; }
        .status-pulse { animation: statusPulse 2.2s ease infinite; }
        textarea { resize: none; }
        textarea::placeholder { color: #c4bdb5; }
        input::placeholder { color: #c4bdb5; }
        button { cursor: pointer; }
        button:disabled { cursor: default; }
      `}</style>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          height: "100vh",
          background: "#f5f2ed",
          color: "#3a3530",
          fontFamily: "'DM Sans', sans-serif",
          fontWeight: 300,
          overflow: "hidden",
        }}
      >
        {/* ── Header ── */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 32px",
            height: "58px",
            borderBottom: "1px solid #e4ddd5",
            flexShrink: 0,
            background: "#f0ece6",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <div
              className={wsStatus !== "connected" ? "status-pulse" : ""}
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                background: statusColor,
                flexShrink: 0,
              }}
            />
            <span
              style={{
                fontFamily: "'Cormorant Garamond', serif",
                fontSize: "20px",
                fontWeight: 300,
                letterSpacing: "0.3em",
                color: "#1c1a17",
                textTransform: "uppercase",
              }}
            >
              Airi
            </span>
            <span
              style={{ width: "1px", height: "14px", background: "#ddd7cf" }}
            />
            <span
              style={{
                fontSize: "10px",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                color: "#b0a89e",
                fontWeight: 400,
              }}
            >
              {statusLabel}
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
            <span
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: "10px",
                color: "#c4bdb5",
                letterSpacing: "0.04em",
              }}
            >
              {sessionId.slice(0, 16)}
            </span>
            <button
              onClick={() => setShowConfig((v) => !v)}
              style={{
                background: "none",
                border: "none",
                fontSize: "10px",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: showConfig ? AMBER : "#b8b0a6",
                padding: "4px 0",
                fontFamily: "'DM Sans', sans-serif",
                fontWeight: 400,
                transition: "color 0.2s",
              }}
            >
              {showConfig ? "Close" : "Config"}
            </button>
          </div>
        </header>

        {/* ── Config bar ── */}
        {showConfig && (
          <div
            style={{
              padding: "10px 32px",
              background: "#ece8e2",
              borderBottom: "1px solid #e4ddd5",
              display: "flex",
              alignItems: "center",
              gap: "16px",
            }}
          >
            <span
              style={{
                fontSize: "9px",
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: AMBER_DIM,
                whiteSpace: "nowrap",
                fontWeight: 400,
              }}
            >
              Endpoint
            </span>
            <input
              type="text"
              value={wsUrl}
              onChange={(e) => setWsUrl(e.target.value)}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                borderBottom: "1px solid #d0c9c0",
                color: "#5a5248",
                fontSize: "11px",
                fontFamily: "'JetBrains Mono', monospace",
                padding: "4px 0",
                outline: "none",
                letterSpacing: "0.03em",
              }}
            />
          </div>
        )}

        {/* ── Messages ── */}
        <div style={{ flex: 1, overflowY: "auto", padding: "48px 0 24px" }}>
          {messages.length === 0 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                userSelect: "none",
              }}
            >
              <div
                style={{
                  fontFamily: "'Cormorant Garamond', serif",
                  fontSize: "80px",
                  fontWeight: 300,
                  letterSpacing: "0.3em",
                  color: "#1c1a17",
                  opacity: 0.06,
                  textTransform: "uppercase",
                  lineHeight: 1,
                }}
              >
                AIRI
              </div>
              <div
                style={{
                  marginTop: "20px",
                  fontSize: "10px",
                  letterSpacing: "0.2em",
                  textTransform: "uppercase",
                  color: "#c4bdb5",
                  fontWeight: 400,
                }}
              >
                Agent · Ready
              </div>
            </div>
          )}

          <div
            style={{ maxWidth: "720px", margin: "0 auto", padding: "0 32px" }}
          >
            {messages.map((msg, idx) => (
              <div
                key={msg.id}
                className="msg-anim"
                style={{
                  marginBottom: "36px",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                <div
                  style={{
                    fontSize: "9px",
                    letterSpacing: "0.2em",
                    textTransform: "uppercase",
                    fontWeight: 400,
                    color:
                      msg.role === "user"
                        ? "#c4bdb5"
                        : msg.role === "error"
                          ? "#c07070"
                          : AMBER_DIM,
                    marginBottom: "10px",
                  }}
                >
                  {msg.role === "user"
                    ? "You"
                    : msg.role === "error"
                      ? "Error"
                      : "Airi"}
                </div>

                {msg.role === "user" ? (
                  <div
                    style={{
                      maxWidth: "68%",
                      background: "#ece8e2",
                      border: "1px solid #e0d9d1",
                      borderRadius: "3px",
                      padding: "14px 20px",
                      fontSize: "14px",
                      lineHeight: "1.8",
                      color: "#2e2b27",
                      whiteSpace: "pre-wrap",
                      letterSpacing: "0.01em",
                    }}
                  >
                    {msg.content}
                  </div>
                ) : (
                  <div
                    style={{
                      width: "100%",
                      borderLeft: `1px solid ${msg.role === "error" ? "#e0b0b0" : "#d8d0c8"}`,
                      paddingLeft: "22px",
                      fontSize: "14.5px",
                      lineHeight: "1.9",
                      color: msg.role === "error" ? "#a04040" : "#4a4540",
                      whiteSpace: "pre-wrap",
                      letterSpacing: "0.005em",
                    }}
                  >
                    {formatMessage(msg.content)}
                    {msg.role === "ai" &&
                      isGenerating &&
                      idx === messages.length - 1 && (
                        <span
                          style={{
                            display: "inline-block",
                            width: "1.5px",
                            height: "13px",
                            background: AMBER,
                            marginLeft: "2px",
                            verticalAlign: "middle",
                            animation: "blink 1s step-end infinite",
                          }}
                        />
                      )}
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Input ── */}
        <div
          style={{
            padding: "16px 32px 28px",
            borderTop: "1px solid #e4ddd5",
            flexShrink: 0,
            background: "#f0ece6",
          }}
        >
          <div style={{ maxWidth: "720px", margin: "0 auto" }}>
            <div
              style={{
                display: "flex",
                gap: "14px",
                alignItems: "flex-end",
                background: "#f5f2ed",
                border: `1px solid ${inputFocused ? "#c4bdb5" : "#ddd7cf"}`,
                borderRadius: "3px",
                padding: "12px 16px",
                transition: "border-color 0.25s",
              }}
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={() => setInputFocused(true)}
                onBlur={() => setInputFocused(false)}
                placeholder="Message Airi…"
                rows={1}
                style={{
                  flex: 1,
                  background: "transparent",
                  border: "none",
                  color: "#2e2b27",
                  fontSize: "14px",
                  lineHeight: "1.65",
                  fontFamily: "'DM Sans', sans-serif",
                  fontWeight: 300,
                  maxHeight: "180px",
                  overflowY: "auto",
                  padding: 0,
                  outline: "none",
                  letterSpacing: "0.01em",
                }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height =
                    Math.min(e.target.scrollHeight, 180) + "px";
                }}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || wsStatus !== "connected"}
                style={{
                  background: "none",
                  border: "none",
                  padding: "2px 4px",
                  color:
                    input.trim() && wsStatus === "connected"
                      ? AMBER
                      : "#d0c9c0",
                  fontSize: "20px",
                  lineHeight: 1,
                  flexShrink: 0,
                  transition: "color 0.2s",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                ↑
              </button>
            </div>

            <div
              style={{
                marginTop: "10px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span
                style={{
                  fontSize: "9.5px",
                  letterSpacing: "0.1em",
                  color: "#c4bdb5",
                  textTransform: "uppercase",
                }}
              >
                Shift + Enter for newline
              </span>
              <span
                style={{
                  fontSize: "9.5px",
                  letterSpacing: "0.1em",
                  color: "#c4bdb5",
                  textTransform: "uppercase",
                }}
              >
                System access enabled
              </span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
