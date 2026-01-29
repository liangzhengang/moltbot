import { createSubsystemLogger } from "../logging/subsystem.js";
import { isTruthyEnvValue } from "../infra/env.js";

const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434";
const DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text";

const debugEmbeddings = isTruthyEnvValue(process.env.CLAWDBOT_DEBUG_MEMORY_EMBEDDINGS);
const log = createSubsystemLogger("memory/embeddings");

const debugLog = (message: string, meta?: unknown) => {
  if (!debugEmbeddings) return;
  const suffix = meta ? ` ${JSON.stringify(meta)}` : "";
  log.raw(`${message}${suffix}`);
};

function normalizeOllamaModel(model: string): string {
  const trimmed = model.trim();
  if (!trimmed) return DEFAULT_OLLAMA_EMBEDDING_MODEL;
  if (trimmed.startsWith("ollama/")) return trimmed.slice("ollama/".length);
  return trimmed;
}

function normalizeOllamaBaseUrl(raw: string | undefined): string {
  if (!raw) return DEFAULT_OLLAMA_BASE_URL;
  return raw.replace(/\/+$/, "");
}

export async function createOllamaEmbeddingProvider(options: {
  remote?: { baseUrl?: string; model?: string; options?: Record<string, unknown> };
  config?: {
    models?: {
      providers?: {
        ollama?: { baseUrl?: string; model?: string; options?: Record<string, unknown> };
      };
    };
  };
  model?: string;
}) {
  const client = await resolveOllamaEmbeddingClient(options);

  const baseUrl = client.baseUrl.replace(/\/$/, "");
  const embedUrl = `${baseUrl}/api/embeddings`;

  const embedQuery = async (text: string): Promise<number[]> => {
    if (!text.trim()) return [];
    const res = await fetch(embedUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: client.model,
        prompt: text,
        options: client.options,
      }),
    });
    if (!res.ok) {
      const payload = await res.text();
      throw new Error(`ollama embeddings failed: ${res.status} ${payload}`);
    }
    const payload = await res.json();
    return payload.embedding ?? [];
  };

  const embedBatch = async (texts: string[]): Promise<number[][]> => {
    if (texts.length === 0) return [];
    const embeddings = await Promise.all(
      texts.map(async (text) => {
        const res = await fetch(embedUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: client.model,
            prompt: text,
            options: client.options,
          }),
        });
        if (!res.ok) {
          const payload = await res.text();
          throw new Error(`ollama embeddings failed: ${res.status} ${payload}`);
        }
        const payload = await res.json();
        return payload.embedding ?? [];
      }),
    );
    return embeddings;
  };

  return {
    provider: {
      id: "ollama",
      model: client.model,
      embedQuery,
      embedBatch,
    },
    client,
  };
}

export async function resolveOllamaEmbeddingClient(options: {
  remote?: { baseUrl?: string; model?: string; options?: Record<string, unknown> };
  config?: {
    models?: {
      providers?: {
        ollama?: { baseUrl?: string; model?: string; options?: Record<string, unknown> };
      };
    };
  };
  model?: string;
}) {
  const remote = options.remote;
  const remoteBaseUrl = remote?.baseUrl?.trim();
  const remoteModel = remote?.model?.trim();
  const remoteOptions = remote?.options;

  const providerConfig = options.config?.models?.providers?.ollama;
  const baseUrl = normalizeOllamaBaseUrl(remoteBaseUrl || providerConfig?.baseUrl);

  const model = normalizeOllamaModel(
    remoteModel || options.model || providerConfig?.model || DEFAULT_OLLAMA_EMBEDDING_MODEL,
  );

  const optionsOverride = remoteOptions || providerConfig?.options;

  debugLog("memory embeddings: ollama client", {
    baseUrl,
    model,
    options: optionsOverride,
  });

  return { baseUrl, model, options: optionsOverride };
}
