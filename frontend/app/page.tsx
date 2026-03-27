"use client";

import { ChangeEvent, useState } from "react";

type PredictionResult = {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
};

type ApiResponse = {
  filename: string;
  content_type: string;
  original_model: PredictionResult;
  mobile_model: PredictionResult;
  models_agree: boolean;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

function formatPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function topClasses(probabilities: Record<string, number>) {
  return Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setError(null);

    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  }

  async function handleSubmit() {
    if (!selectedFile) {
      setError("Choose an MRI image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? "Prediction failed.");
      }

      setResult(payload);
    } catch (submissionError) {
      const message =
        submissionError instanceof Error
          ? submissionError.message
          : "Something went wrong while talking to the backend.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Clinical Imaging Assistant</p>
          <h1>Upload an MRI and get a fast tumor screening prediction.</h1>
          <p className="hero-text">
            Compare the original DenseNet checkpoint with the mobile Lite export
            through the FastAPI backend in one clean workflow.
          </p>
          <div className="hero-badges">
            <span>FastAPI backend</span>
            <span>Original + mobile model</span>
            <span>MRI upload preview</span>
          </div>
        </div>

        <div className="hero-panel">
          <div className="upload-card">
            <label className="upload-zone">
              <input
                accept="image/*"
                className="hidden-input"
                type="file"
                onChange={handleFileChange}
              />
              {previewUrl ? (
                <img className="preview-image" src={previewUrl} alt="MRI preview" />
              ) : (
                <div className="upload-placeholder">
                  <strong>Select MRI image</strong>
                  <span>JPEG or PNG from your local device</span>
                </div>
              )}
            </label>

            <div className="upload-meta">
              <div>
                <p className="meta-label">Selected file</p>
                <p className="meta-value">
                  {selectedFile ? selectedFile.name : "No file selected"}
                </p>
              </div>
              <button
                className="predict-button"
                onClick={handleSubmit}
                disabled={loading}
                type="button"
              >
                {loading ? "Analyzing MRI..." : "Run Prediction"}
              </button>
            </div>

            {error ? <p className="error-banner">{error}</p> : null}
          </div>
        </div>
      </section>

      <section className="results-grid">
        <article className="status-card">
          <p className="section-label">Backend Target</p>
          <h2>{API_BASE_URL}</h2>
          <p>
            Make sure the FastAPI server is running before uploading an image.
          </p>
        </article>

        <article className="status-card accent-card">
          <p className="section-label">Screening Summary</p>
          <h2>
            {result
              ? result.original_model.label === "No-tumor"
                ? "No tumor detected by primary model"
                : `${result.original_model.label} detected`
              : "Awaiting MRI upload"}
          </h2>
          <p>
            {result
              ? result.models_agree
                ? "Both models agree on the top prediction."
                : "The two models disagree. Review with extra care."
              : "Once you upload an MRI, the page will summarize the outcome here."}
          </p>
        </article>
      </section>

      {result ? (
        <section className="comparison-section">
          <div className="comparison-header">
            <p className="section-label">Model Comparison</p>
            <h2>Original checkpoint vs mobile Lite model</h2>
          </div>

          <div className="model-grid">
            {[
              ["Original Model", result.original_model],
              ["Mobile Lite Model", result.mobile_model],
            ].map(([title, modelResult]) => (
              <article className="model-card" key={title}>
                <div className="model-card-top">
                  <p className="model-title">{title}</p>
                  <span className="confidence-pill">
                    {formatPercent((modelResult as PredictionResult).confidence)}
                  </span>
                </div>
                <h3>{(modelResult as PredictionResult).label}</h3>
                <div className="bars">
                  {topClasses((modelResult as PredictionResult).probabilities).map(
                    ([label, score]) => (
                      <div className="bar-row" key={label}>
                        <div className="bar-copy">
                          <span>{label}</span>
                          <span>{formatPercent(score)}</span>
                        </div>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${Math.max(score * 100, 2)}%` }}
                          />
                        </div>
                      </div>
                    ),
                  )}
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}
