import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  fallbackReasoningLevelFromProviderError,
  streamWithProvider,
} from "../llm/providers";

const { streamText, responses } = vi.hoisted(() => ({
  streamText: vi.fn((_options: Record<string, unknown>) => ({
    stream: (async function* () {})(),
  })),
  responses: vi.fn((model: string) => ({ modelId: model })),
}));

vi.mock("ai", () => ({
  streamText,
  stepCountIs: (count: number) => count,
}));

vi.mock("@ai-sdk/openai", () => ({
  createOpenAI: () => ({ responses }),
}));

describe("streamWithProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    process.env.OPENAI_API_KEY = "test-key";
  });

  // Zero Data Retention organizations reject replayed item ids, which breaks
  // every tool step after the first.
  it("keeps OpenAI Responses calls free of stored-item references", async () => {
    await streamWithProvider({
      model: "gpt-5.6-sol",
      systemPrompt: "system",
      messages: [{ role: "user", content: "hello" }],
    });

    expect(streamText).toHaveBeenCalledTimes(1);
    expect(streamText.mock.calls[0][0].providerOptions).toEqual({
      openai: { store: false, include: ["reasoning.encrypted_content"] },
    });
  });
});

describe("fallbackReasoningLevelFromProviderError", () => {
  it("selects the nearest level advertised by a provider", () => {
    const error = new Error(
      "Unsupported value: 'low' is not supported with the model. Supported values are: 'none', 'medium', 'high', and 'xhigh'.",
    );

    expect(fallbackReasoningLevelFromProviderError(error, "low")).toBe(
      "medium",
    );
  });

  it("does not retry unrelated provider failures", () => {
    expect(
      fallbackReasoningLevelFromProviderError(
        new Error("The provider is unavailable"),
        "high",
      ),
    ).toBeUndefined();
  });
});
