import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { generateReply, parseGeneration } from './generate'
import { AiError, type AiConfig } from './types'
import { canReuseStoredAiKey } from './stored-key'

function config(overrides: Partial<AiConfig> = {}): AiConfig {
  return {
    provider: 'openai',
    model: 'gpt-test',
    apiKey: 'sk-test',
    baseUrl: null,
    systemPrompt: null,
    isActive: true,
    autoReplyEnabled: false,
    autoReplyMaxPerConversation: 3,
    handoffAgentId: null,
    embeddingsApiKey: null,
    ...overrides,
  }
}

function okResponse(json: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => json,
  } as unknown as Response
}

function errResponse(status: number, json: unknown): Response {
  return {
    ok: false,
    status,
    json: async () => json,
  } as unknown as Response
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
})
afterEach(() => vi.unstubAllGlobals())

describe('parseGeneration', () => {
  it('returns text with no handoff', () => {
    expect(parseGeneration('Hello there')).toEqual({
      text: 'Hello there',
      handoff: false,
      usage: null,
    })
  })

  it('detects + strips the handoff sentinel', () => {
    expect(parseGeneration('[[HANDOFF]]')).toEqual({
      text: '',
      handoff: true,
      usage: null,
    })
    expect(parseGeneration('Let me get a human [[HANDOFF]]')).toEqual({
      text: 'Let me get a human',
      handoff: true,
      usage: null,
    })
  })

  it('passes usage straight through', () => {
    const usage = { promptTokens: 10, completionTokens: 5, totalTokens: 15 }
    expect(parseGeneration('Hi', usage)).toEqual({
      text: 'Hi',
      handoff: false,
      usage,
    })
  })
})

describe('generateReply — OpenAI', () => {
  it('calls the chat completions endpoint and returns the reply', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        choices: [{ message: { content: 'Sure — happy to help!' } }],
        usage: { prompt_tokens: 42, completion_tokens: 8, total_tokens: 50 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({ provider: 'openai' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(res).toEqual({
      text: 'Sure — happy to help!',
      handoff: false,
      usage: { promptTokens: 42, completionTokens: 8, totalTokens: 50 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toContain('api.openai.com')
    expect(opts.headers.Authorization).toBe('Bearer sk-test')
  })

  it('maps a 401 to an invalid_key AiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(401, { error: { message: 'Incorrect API key' } }),
      ),
    )

    await expect(
      generateReply({
        config: config(),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })
  })

  it('throws on an empty completion', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ choices: [{ message: { content: '' } }] })),
    )
    await expect(
      generateReply({
        config: config(),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toBeInstanceOf(AiError)
  })
})

describe('generateReply — Anthropic', () => {
  it('calls the messages endpoint with the version header and parses text blocks', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        content: [{ type: 'text', text: 'Hi there!' }],
        usage: { input_tokens: 30, output_tokens: 6 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({ provider: 'anthropic', apiKey: 'sk-ant-x' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hello' }],
    })

    // Anthropic reports input/output only — total is summed by normalizeUsage.
    expect(res).toEqual({
      text: 'Hi there!',
      handoff: false,
      usage: { promptTokens: 30, completionTokens: 6, totalTokens: 36 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toContain('api.anthropic.com')
    expect(opts.headers['x-api-key']).toBe('sk-ant-x')
    expect(opts.headers['anthropic-version']).toBeTruthy()
  })

  it('detects handoff in the model output', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({ content: [{ type: 'text', text: '[[HANDOFF]]' }] }),
      ),
    )
    const res = await generateReply({
      config: config({ provider: 'anthropic' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'I want to speak to a person' }],
    })
    expect(res.handoff).toBe(true)
    expect(res.text).toBe('')
  })

  it('drops a leading assistant turn so the payload starts on the customer', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(okResponse({ content: [{ type: 'text', text: 'ok' }] }))
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'anthropic' }),
      systemPrompt: 'sys',
      messages: [
        { role: 'assistant', content: 'Welcome!' },
        { role: 'user', content: 'Hi' },
      ],
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body.messages[0].role).toBe('user')
    expect(body.messages).toHaveLength(1)
  })
})


describe('generateReply — OpenAI-compatible', () => {
  it('uses the configured endpoint and max_tokens', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: 'Gateway reply' } }],
          usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    const result = await generateReply({
      config: config({
        provider: 'openai_compatible',
        model: 'llama-3.3-70b',
        baseUrl: 'https://8.8.8.8/v1/',
      }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(result).toEqual({
      text: 'Gateway reply',
      handoff: false,
      usage: { promptTokens: 12, completionTokens: 4, totalTokens: 16 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toBe('https://8.8.8.8/v1/chat/completions')
    expect(opts.headers.Authorization).toBe('Bearer sk-test')
    expect(JSON.parse(opts.body).max_tokens).toBe(1024)
    expect(opts.redirect).toBe('manual')
  })

  it('blocks private hosts unless the operator explicitly allow-lists them', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const previous = process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS
    delete process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS
    try {
      await expect(
        generateReply({
          config: config({
            provider: 'openai_compatible',
            baseUrl: 'https://127.0.0.1:11434/v1',
          }),
          systemPrompt: 'sys',
          messages: [{ role: 'user', content: 'Hi' }],
        }),
      ).rejects.toMatchObject({ code: 'blocked_url', status: 400 })
      expect(fetchMock).not.toHaveBeenCalled()
    } finally {
      if (previous === undefined) delete process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS
      else process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS = previous
    }
  })

  it('allows an operator-approved private host', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: 'Local reply' } }],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    const previous = process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS
    process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS = '127.0.0.1'
    try {
      const result = await generateReply({
        config: config({
          provider: 'openai_compatible',
          model: 'llama3.2',
          baseUrl: 'http://127.0.0.1:11434/v1',
        }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      })
      expect(result.text).toBe('Local reply')
      expect(fetchMock).toHaveBeenCalledOnce()
    } finally {
      if (previous === undefined) delete process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS
      else process.env.AI_PROVIDER_ALLOWED_PRIVATE_HOSTS = previous
    }
  })

  it('does not follow redirects', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 302 }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      generateReply({
        config: config({
          provider: 'openai_compatible',
          baseUrl: 'https://8.8.8.8/v1',
        }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'provider_error' })

    expect(fetchMock.mock.calls[0][1].redirect).toBe('manual')
  })
})

describe('generateReply — Gemini', () => {
  it('calls the generateContent endpoint with x-goog-api-key and returns the reply', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        candidates: [{ content: { parts: [{ text: 'Hello from Gemini!' }] } }],
        usageMetadata: {
          promptTokenCount: 20,
          candidatesTokenCount: 5,
          totalTokenCount: 25,
        },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({
        provider: 'gemini',
        model: 'gemini-3.5-flash-lite',
        apiKey: 'AIza-test-key-123',
      }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(res).toEqual({
      text: 'Hello from Gemini!',
      handoff: false,
      usage: { promptTokens: 20, completionTokens: 5, totalTokens: 25 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toContain('generativelanguage.googleapis.com')
    expect(url).toContain('gemini-3.5-flash-lite')
    expect(url).toContain(':generateContent')
    // Authentication header assertions:
    expect(opts.headers['x-goog-api-key']).toBe('AIza-test-key-123')
    expect(opts.headers.Authorization).toBeUndefined()
  })

  it('supports new AQ. API key format unchanged without modifying or truncating', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        candidates: [{ content: { parts: [{ text: 'Key valid!' }] } }],
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const aqKey = 'AQ.AbCdEf1234567890_CustomFormatKeyLong'
    await generateReply({
      config: config({
        provider: 'gemini',
        model: 'gemini-3.5-flash-lite',
        apiKey: aqKey,
      }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Ping' }],
    })

    const [, opts] = fetchMock.mock.calls[0]
    expect(opts.headers['x-goog-api-key']).toBe(aqKey)
    expect(opts.headers.Authorization).toBeUndefined()
  })

  it('rejects an empty API key appropriately without calling fetch', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite', apiKey: '' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })

    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('sends system_instruction and maps assistant → model role', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ candidates: [{ content: { parts: [{ text: 'ok' }] } }] }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
      systemPrompt: 'Be helpful',
      messages: [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there!' },
        { role: 'user', content: 'How are you?' },
      ],
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body.system_instruction.parts[0].text).toBe('Be helpful')
    // assistant → model
    expect(body.contents[1].role).toBe('model')
    expect(body.contents[0].role).toBe('user')
  })

  it('drops a leading assistant turn so the payload starts on the customer', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ candidates: [{ content: { parts: [{ text: 'ok' }] } }] }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
      systemPrompt: 'sys',
      messages: [
        { role: 'assistant', content: 'Welcome!' },
        { role: 'user', content: 'Hi' },
      ],
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body.contents[0].role).toBe('user')
    expect(body.contents).toHaveLength(1)
  })

  it('merges consecutive messages of the same role', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ candidates: [{ content: { parts: [{ text: 'ok' }] } }] }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
      systemPrompt: 'sys',
      messages: [
        { role: 'user', content: 'First line' },
        { role: 'user', content: 'Second line' },
      ],
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body.contents).toHaveLength(1)
    expect(body.contents[0].role).toBe('user')
    expect(body.contents[0].parts[0].text).toBe('First line\n\nSecond line')
  })

  it('detects handoff sentinel in Gemini output', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({
          candidates: [{ content: { parts: [{ text: '[[HANDOFF]]' }] } }],
        }),
      ),
    )
    const res = await generateReply({
      config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'I need a human' }],
    })
    expect(res.handoff).toBe(true)
    expect(res.text).toBe('')
  })

  it('maps a 401 to an invalid_key AiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(401, { error: { message: 'API key not valid.' } }),
      ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })
  })

  it('maps a 400 with API key error message to an invalid_key AiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(400, {
          error: {
            code: 400,
            message: 'API key not valid. Please pass a valid API key.',
            status: 'INVALID_ARGUMENT',
          },
        }),
      ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })
  })

  it('maps a 404 with invalid model to provider_error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(404, {
          error: {
            code: 404,
            message: 'models/invalid-model is not found for API version v1beta',
            status: 'NOT_FOUND',
          },
        }),
      ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'invalid-model' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'provider_error', status: 502 })
  })

  it('maps a 429 to a rate_limited AiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(errResponse(429, { error: { message: 'quota exceeded' } })),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'rate_limited' })
  })

  it('handles network timeout gracefully', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new DOMException('The operation was aborted.', 'TimeoutError')),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'timeout', status: 504 })
  })

  it('handles safety blocked response gracefully via promptFeedback', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({
          promptFeedback: { blockReason: 'SAFETY' },
        }),
      ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Blocked content' }],
      }),
    ).rejects.toMatchObject({ code: 'safety_blocked', status: 502 })
  })

  it('handles safety blocked response gracefully via finishReason', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({
          candidates: [
            {
              finishReason: 'SAFETY',
              content: { parts: [] },
            },
          ],
        }),
      ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Blocked content' }],
      }),
    ).rejects.toMatchObject({ code: 'safety_blocked', status: 502 })
  })

  it('throws an empty_response AiError when candidates array is empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ candidates: [] })),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'empty_response' })
  })

  it('handles unexpected or malformed API response structure without crashing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ unexpected: { random: 123 } })),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'empty_response' })
  })

  it('redacts API key from error detail if present', async () => {
    const secretKey = 'AQ.SecretKey123'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(500, {
          error: {
            message: `Internal error processing key ${secretKey}`,
          },
        }),
      ),
    )
    const promise = generateReply({
      config: config({ provider: 'gemini', model: 'gemini-3.5-flash-lite', apiKey: secretKey }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    await expect(promise).rejects.toMatchObject({ code: 'provider_error' })
    try {
      await promise
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : ''
      expect(msg).toContain('[REDACTED]')
      expect(msg).not.toContain(secretKey)
    }
  })
})

describe('Provider switching and credential reuse', () => {
  it('does not reuse stored credentials across different providers', () => {
    const openaiStored = { provider: 'openai' as const, base_url: null }
    const anthropicStored = { provider: 'anthropic' as const, base_url: null }
    const geminiStored = { provider: 'gemini' as const, base_url: null }

    // Gemini cannot reuse OpenAI or Anthropic keys
    expect(canReuseStoredAiKey(openaiStored, 'gemini', null)).toBe(false)
    expect(canReuseStoredAiKey(anthropicStored, 'gemini', null)).toBe(false)

    // OpenAI cannot reuse Gemini keys
    expect(canReuseStoredAiKey(geminiStored, 'openai', null)).toBe(false)

    // Anthropic cannot reuse Gemini keys
    expect(canReuseStoredAiKey(geminiStored, 'anthropic', null)).toBe(false)

    // Same provider can reuse
    expect(canReuseStoredAiKey(geminiStored, 'gemini', null)).toBe(true)
  })
})
