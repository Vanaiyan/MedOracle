/**
 * src/components/ChatbotPanel.jsx
 * LLM chat interface with auto-explanation + follow-up Q&A.
 */

import { useState, useEffect, useRef } from 'react';
import { chatAPI } from '../api/client';

function renderMarkdown(text) {
  const lines = text.split('\n');
  const elements = [];
  let listItems = [];

  const flushList = () => {
    if (listItems.length) {
      elements.push(
        <ul key={`ul-${elements.length}`} style={{ margin: '0.4rem 0 0.4rem 1.1rem', padding: 0 }}>
          {listItems.map((item, i) => <li key={i} style={{ marginBottom: 2 }}>{inlineFormat(item)}</li>)}
        </ul>
      );
      listItems = [];
    }
  };

  const inlineFormat = (str) => {
    // **bold** and *italic*
    const parts = str.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**'))
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      if (part.startsWith('*') && part.endsWith('*'))
        return <em key={i}>{part.slice(1, -1)}</em>;
      return part;
    });
  };

  lines.forEach((line, i) => {
    const bullet = line.match(/^[-*]\s+(.+)/);
    if (bullet) {
      listItems.push(bullet[1]);
    } else {
      flushList();
      if (line.trim() === '') {
        if (elements.length) elements.push(<br key={`br-${i}`} />);
      } else {
        elements.push(<span key={`l-${i}`} style={{ display: 'block' }}>{inlineFormat(line)}</span>);
      }
    }
  });
  flushList();
  return elements;
}

function TypingIndicator() {
  return (
    <div className="chat-bubble typing">
      <div style={{ display: 'flex', gap: 4, alignItems: 'center', padding: '2px 0' }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: 6, height: 6, borderRadius: '50%',
            background: 'var(--accent-blue)',
            animation: `typingDot 1.2s ${i * 0.2}s ease infinite`,
          }} />
        ))}
      </div>
      <style>{`
        @keyframes typingDot {
          0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
          40%            { opacity: 1;   transform: scale(1);   }
        }
      `}</style>
    </div>
  );
}

export default function ChatbotPanel({ sessionId, autoExplanation, onClose }) {
  const [messages, setMessages] = useState([]);
  const [input,    setInput   ] = useState('');
  const [sending,  setSending ] = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  // Load history + inject auto-explanation on session change
  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      return;
    }
    chatAPI.history(sessionId).then(r => {
      const hist = r.data.messages || [];
      if (hist.length === 0 && autoExplanation) {
        setMessages([{
          id: 'auto',
          role: 'assistant',
          content: autoExplanation,
        }]);
      } else {
        setMessages(hist.map(m => ({ id: m.message_id, role: m.role, content: m.content })));
      }
    }).catch(() => {
      if (autoExplanation) {
        setMessages([{ id: 'auto', role: 'assistant', content: autoExplanation }]);
      }
    });
  }, [sessionId, autoExplanation]);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || !sessionId || sending) return;

    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setSending(true);

    try {
      const { data } = await chatAPI.send(sessionId, text);
      setMessages(prev => [...prev, { id: data.message_id, role: 'assistant', content: data.response }]);
    } catch {
      setMessages(prev => [...prev, {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: '⚠️ Failed to get a response. Please try again.',
      }]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="chatbot-panel">
      {/* Header */}
      <div className="chat-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: 'var(--gradient-btn)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14,
          }}>🤖</div>
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 700 }}>MedOracle Assistant</h3>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: 0 }}>
              {sessionId ? 'Ask about this session' : 'Select a session to start'}
            </p>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--accent-green)' }} />
              <span style={{ fontSize: 11, color: 'var(--accent-green)' }}>AI Active</span>
            </div>
            {onClose && (
              <button onClick={onClose} style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: 'var(--text-muted)', fontSize: 16, lineHeight: 1,
                padding: '2px 4px', borderRadius: 4,
                transition: 'color 0.15s',
              }}
                onMouseEnter={e => e.target.style.color = 'var(--text-primary)'}
                onMouseLeave={e => e.target.style.color = 'var(--text-muted)'}
                title="Close"
              >✕</button>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {!sessionId ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, color: 'var(--text-muted)', fontSize: 13, textAlign: 'center', padding: '1rem' }}>
            <span style={{ fontSize: 32 }}>💬</span>
            <p>Select a session from the history panel,<br />or run a new prediction to get an AI explanation.</p>
          </div>
        ) : messages.length === 0 ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '1rem' }}>
            <span className="spinner" />
          </div>
        ) : (
          messages.map(m => (
            <div key={m.id} className={`chat-bubble ${m.role}`}>
              {m.role === 'assistant' ? renderMarkdown(m.content) : m.content}
            </div>
          ))
        )}
        {sending && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          id="chat-input"
          className="chat-input"
          rows={1}
          placeholder={sessionId ? "Ask a follow-up question…" : "Select a session first"}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!sessionId || sending}
          style={{ minHeight: 38, maxHeight: 100 }}
        />
        <button
          id="chat-send-btn"
          className="chat-send-btn"
          onClick={send}
          disabled={!input.trim() || !sessionId || sending}
          title="Send"
        >
          {sending ? <span className="spinner" style={{ width: 14, height: 14 }} /> : '↑'}
        </button>
      </div>
    </div>
  );
}
