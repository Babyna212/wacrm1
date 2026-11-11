import { AiError, type ChatMessage, type ProviderResult } from '../types'
import { MAX_OUTPUT_TOKENS } from '../defaults'
import { mergeConsecutive, normalizeUsage, toNetworkError, type ProviderArgs } from './shared'
import { getT } from '@/lib/i18n/translate'

const t = getT('LibErrors')

// ============================================================
// Google Gemini (generateContent) adapter.
//
// Uses the Gemini Developer API v1beta endpoint. The caller
// supplies their own API key (Bearer token via Authorization
// header, which works for both API-key and OAuth tokens).
//
// Mapping:
//   system prompt  → system_instruction (role: "system")
//   chat history   → contents[] (role: "user" | "model")
//   response       → candidates[0].content.parts[0].text
//   usage          → usageMetadata (promptTokenCount / candidatesTokenCount)
//
// Role normalisation: Gemini uses "model" for assistant turns;
// we map our internal "assistant" → "model" here.
// ============================================================

const GEMINI_API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models'

interface GeminiPart {
  text?: string
}

interface GeminiContent {
  role: 'user' | 'model'
  parts: GeminiPart[]
}

interface GeminiResponse {
  candidates?: {
    content?: {
      parts?: GeminiPart[]
    }
  }[]
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
  }
}

/**
 * Normalise our internal ChatMessage history into Gemini's
 * `contents` array. Gemini requires strictly alternating
 * user/model turns starting with "user". Collapses consecutive
 * same-role turns (via `mergeConsecutive`) and drops any leading
 * "model" turns that would appear before the first user turn.
 */
function toGeminiContents(messages: ChatMessage[]): GeminiContent[] {
  const merged = mergeConsecutive(messages)
  // Strip leading assistant turns — Gemini requires user-first.
  while (merged.length > 0 && merged[0].role === 'assistant') {
    merged.shift()
  }
  if (merged.length === 0) {
    return [{ role: 'user', parts: [{ text: '(The customer has not sent a message yet.)' }] }]
  }
  return merged.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }))
}

/**
 * Call the Gemini generateContent endpoint with the caller's own
 * API key. Returns the raw assistant text + token usage (handoff
 * parsing happens in `generateReply`).
 *
 * The endpoint pattern is:
 *   POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
 *   Authorization: Bearer {apiKey}
 */
export async function generateGemini(args: ProviderArgs): Promise<ProviderResult> {
  const { apiKey, model, systemPrompt, messages, timeoutMs } = args

  const endpoint = `${GEMINI_API_BASE}/${encodeURIComponent(model)}:generateContent`

  let res: Response
  try {
    res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        system_instruction: {
          parts: [{ text: systemPrompt }],
        },
        contents: toGeminiContents(messages),
        generationConfig: {
          maxOutputTokens: MAX_OUTPUT_TOKENS,
        },
      }),
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (err) {
    throw toNetworkError(err)
  }

  if (!res.ok) {
    // Gemini error shape: { error: { code, message, status } }
    let detail = ''
    try {
      const body = (await res.json()) as { error?: { message?: string } }
      detail = body?.error?.message ?? ''
    } catch {
      // non-JSON body — fall back to status line
    }

    const { status } = res
    const code =
      status === 400 && detail.toLowerCase().includes('api key')
        ? 'invalid_key'
        : status === 401 || status === 403
          ? 'invalid_key'
          : status === 429
            ? 'rate_limited'
            : 'provider_error'
    const base =
      code === 'invalid_key'
        ? t('ai.invalidKey', { provider: 'Gemini' })
        : code === 'rate_limited'
          ? t('ai.rateLimited', { provider: 'Gemini' })
          : t('ai.apiError', { provider: 'Gemini', status: String(status) })

    throw new AiError(detail ? `${base}: ${detail}` : base, {
      code,
      status: code === 'invalid_key' ? 401 : 502,
    })
  }

  const data = (await res.json().catch(() => null)) as GeminiResponse | null
  const text = data?.candidates?.[0]?.content?.parts
    ?.filter((p) => typeof p.text === 'string')
    .map((p) => p.text)
    .join('')
    .trim()

  if (!text) {
    throw new AiError(t('ai.emptyResponse', { provider: 'Gemini' }), {
      code: 'empty_response',
    })
  }

  const usage = normalizeUsage({
    prompt: data?.usageMetadata?.promptTokenCount,
    completion: data?.usageMetadata?.candidatesTokenCount,
    total: data?.usageMetadata?.totalTokenCount,
  })

  return { text, usage }
}

