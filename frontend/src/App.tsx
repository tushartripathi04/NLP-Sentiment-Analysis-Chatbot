import { useState } from 'react'
import './App.css'

type SentimentResult = {
  sentiment: 'positive' | 'negative' | 'neutral'
  label: 0 | 1 | 2
  model: string
}

const API_URL = (
  import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

const examples = [
  {
    label: 'Positive',
    text: 'I absolutely love this product. The quality is excellent!',
  },
  {
    label: 'Negative',
    text: 'This was a disappointing experience. The service was terrible.',
  },
  {
    label: 'Neutral',
    text: 'The package arrived today and contains the items I ordered.',
  },
]

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<SentimentResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function analyzeSentiment(input?: string) {
    const content = (input ?? text).trim()

    if (!content) {
      setError('Enter some text before running sentiment analysis.')
      setResult(null)
      return
    }

    setText(content)
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(
        `${API_URL}/api/v1/sentiment/predict`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: content }),
        },
      )

      if (!response.ok) {
        const body = await response.json().catch(() => null)
        const detail = body?.detail
        throw new Error(
          typeof detail === 'string'
            ? detail
            : `Request failed with status ${response.status}.`,
        )
      }

      const data: SentimentResult = await response.json()
      setResult(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message.includes('Failed to fetch')
            ? 'Cannot reach the API. Make sure the FastAPI backend is running and CORS is configured.'
            : err.message
          : 'An unexpected error occurred.',
      )
    } finally {
      setLoading(false)
    }
  }

  const sentiment = result?.sentiment

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" aria-label="SentimentIQ home">
          <span className="brand-mark">S</span>
          <span>
            <strong>SentimentIQ</strong>
            <small>TEXT INTELLIGENCE</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>
        <a className="nav-item active" href="#analyzer">
          <span className="nav-icon">â—‰</span>
          Sentiment analyzer
        </a>
        <a className="nav-item" href="#how-it-works">
          <span className="nav-icon">âŒ˜</span>
          How it works
        </a>

        <div className="sidebar-bottom">
          <span className="status-dot" />
          <div>
            <strong>Local workspace</strong>
            <small>Sentiment API integration</small>
          </div>
        </div>
      </aside>

      <main className="main-content" id="home">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> Sentiment analyzer</div>
          <span className="api-badge">
            <span className="status-dot" />
            ML-powered analysis
          </span>
        </header>

        <section className="page-heading" id="analyzer">
          <div className="eyebrow">NLP Â· TEXT CLASSIFICATION</div>
          <h1>Understand the sentiment<br className="desktop-break" /> behind your words.</h1>
          <p>
            Analyze written text and discover whether its overall sentiment
            is positive, negative, or neutral.
          </p>
        </section>

        <section className="workspace-grid">
          <div className="panel input-panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">01 / INPUT</span>
                <h2>Text analysis</h2>
              </div>
              <span className="soft-tag">English text</span>
            </div>

            <label className="field-label" htmlFor="sentiment-text">
              Your text
            </label>
            <textarea
              id="sentiment-text"
              value={text}
              onChange={(event) => {
                setText(event.target.value)
                setError('')
              }}
              placeholder="Paste a review, message, comment, or any text you want to analyze..."
              maxLength={10000}
              rows={9}
              disabled={loading}
            />

            <div className="input-meta">
              <span>Be specific for a more useful analysis.</span>
              <span>{text.length.toLocaleString()} / 10,000</span>
            </div>

            <div className="examples">
              <span className="field-label">TRY AN EXAMPLE</span>
              <div className="example-buttons">
                {examples.map((example) => (
                  <button
                    className="example-button"
                    type="button"
                    key={example.label}
                    disabled={loading}
                    onClick={() => {
                      setText(example.text)
                      setResult(null)
                      setError('')
                    }}
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              className="analyze-button"
              type="button"
              onClick={() => void analyzeSentiment()}
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analyzing text...
                </>
              ) : (
                <>
                  Analyze sentiment <span aria-hidden="true">â†’</span>
                </>
              )}
            </button>

            <p className="privacy-note">
              <span aria-hidden="true">â—‡</span>
              Analysis runs through your configured SentimentIQ API.
            </p>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <span className="step-label">02 / RESULTS</span>
                <h2>Analysis result</h2>
              </div>
              <span className="result-icon" aria-hidden="true">â†—</span>
            </div>

            {!result && !error && !loading && (
              <div className="empty-result">
                <div className="empty-illustration">
                  <span className="empty-orbit orbit-one" />
                  <span className="empty-orbit orbit-two" />
                  <span className="empty-core">S</span>
                </div>
                <h3>Your result will appear here</h3>
                <p>
                  Enter some text and run the analysis to see the model's
                  predicted sentiment.
                </p>
                <div className="sentiment-legend">
                  <span><i className="legend-dot positive" /> Positive</span>
                  <span><i className="legend-dot neutral" /> Neutral</span>
                  <span><i className="legend-dot negative" /> Negative</span>
                </div>
              </div>
            )}

            {loading && (
              <div className="feedback-state" role="status">
                <span className="large-spinner" />
                <h3>Analyzing your text</h3>
                <p>Sending text to the sentiment prediction API.</p>
              </div>
            )}

            {error && !loading && (
              <div className="error-state" role="alert">
                <span className="error-symbol">!</span>
                <h3>Analysis could not be completed</h3>
                <p>{error}</p>
                <button
                  className="retry-button"
                  type="button"
                  onClick={() => void analyzeSentiment()}
                  disabled={!text.trim()}
                >
                  Try again
                </button>
              </div>
            )}

            {result && sentiment && !loading && (
              <div className="result-content" aria-live="polite">
                <div className={`sentiment-display ${sentiment}`}>
                  <span className="sentiment-symbol" aria-hidden="true">
                    {sentiment === 'positive'
                      ? 'â†—'
                      : sentiment === 'negative'
                        ? 'â†˜'
                        : 'â€”'}
                  </span>
                  <span className="sentiment-caption">PREDICTED SENTIMENT</span>
                  <h3>{sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}</h3>
                  <p>
                    {sentiment === 'positive'
                      ? 'The model classified this text as positive.'
                      : sentiment === 'negative'
                        ? 'The model classified this text as negative.'
                        : 'The model classified this text as neutral.'}
                  </p>
                </div>

                <div className="detail-list">
                  <div className="detail-row">
                    <span>Predicted class</span>
                    <strong>{result.label}</strong>
                  </div>
                  <div className="detail-row">
                    <span>Model</span>
                    <strong>{result.model}</strong>
                  </div>
                  <div className="detail-row">
                    <span>Input length</span>
                    <strong>{text.length.toLocaleString()} characters</strong>
                  </div>
                </div>

                <div className="result-disclaimer">
                  This is a model prediction, not a certainty score. Sarcasm,
                  context, and ambiguous language may affect the result.
                </div>

                <button
                  type="button"
                  className="reset-button"
                  onClick={() => {
                    setText('')
                    setResult(null)
                    setError('')
                  }}
                >
                  Analyze another text <span aria-hidden="true">â†»</span>
                </button>
              </div>
            )}
          </div>
        </section>

        <section className="info-section" id="how-it-works">
          <div>
            <span className="step-label">BUILT WITH NLP</span>
            <h2>From text to insight</h2>
            <p>A simple workflow powered by your trained machine-learning model.</p>
          </div>
          <div className="info-step">
            <span>01</span>
            <strong>Input text</strong>
            <p>Provide a review, comment, or message.</p>
          </div>
          <div className="info-step">
            <span>02</span>
            <strong>TF-IDF features</strong>
            <p>The saved vectorizer transforms the text.</p>
          </div>
          <div className="info-step">
            <span>03</span>
            <strong>Classification</strong>
            <p>LinearSVC predicts the sentiment class.</p>
          </div>
        </section>

        <footer className="footer">
          <span>SentimentIQ <span className="footer-divider">/</span> NLP Sentiment Analysis</span>
          <span>Phase 7 Â· API integration</span>
        </footer>
      </main>
    </div>
  )
}

export default App
