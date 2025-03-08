//
//  SLMEvaluator.swift
//  ai_chat_app
//
//  Created by 池内隆人 on 2025/02/09.
//

import MLX
import MLXRandom
import SwiftUI

@Observable
@MainActor
class SLMEvaluator {

  var running = false

  var output = ""
  var modelInfo = ""
  var stat = ""

  // モデルを設定
  let modelConfiguration = ModelRegistry.phi4_mini_8bit

  /// parameters controlling the output
  let generateParameters = GenerateParameters(temperature: 0)
  let maxTokens = 2000

  /// update the display every N tokens -- 4 looks like it updates continuously
  /// and is low overhead.  observed ~15% reduction in tokens/s when updating
  /// on every token
  let displayEveryNTokens = 4

  enum LoadState {
    case idle
    case loaded(MLXModelContainer)
  }

  var loadState = LoadState.idle

  /// load and return the model -- can be called multiple times, subsequent calls will
  /// just return the loaded model
  func load() async throws -> MLXModelContainer {
    switch loadState {
    case .idle:
      // limit the buffer cache
      MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
      let modelContainer = try await LLMModelFactory.shared.loadContainer(
        configuration: modelConfiguration
      ) {
        [modelConfiguration] progress in
        Task { @MainActor in
          self.modelInfo =
            "Downloading now...: \(Int(progress.fractionCompleted * 100))%"
        }
      }
      let numParams = await modelContainer.perform { context in
        context.model.numParameters()
      }

      self.modelInfo =
        "Standby OK!"
      loadState = .loaded(modelContainer)
      return modelContainer

    case .loaded(let modelContainer):
      return modelContainer
    }
  }

  func generate(prompt: String) async {
    guard !running else { return }

    running = true
    self.output = ""

    do {
      let modelContainer = try await load()
      let systemPrompt = """
        あなたは、ユーザーの多岐にわたる質問やリクエストに対して、できるだけ簡潔に回答するアシスタントです。
        無駄に詳細を述べず、必要最小限の情報だけを提供してください。
        もし答えが分からない場合は、正直に『わかりません』と述べてください。
        """

      // each time you generate you will get something new
      MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
      let result = try await modelContainer.perform { context in
        let input = try await context.processor.prepare(
          input: .init(messages: [
            ["role": "system", "content": systemPrompt], ["role": "user", "content": prompt],
          ]))
        return try mlxGenerate(
          input: input, parameters: generateParameters, context: context
        ) { tokens in
          // update the output -- this will make the view show the text as it generates
          if tokens.count % displayEveryNTokens == 0 {
            let text = context.tokenizer.decode(tokens: tokens)
            Task { @MainActor in
              print(text)
              self.output = text
            }
          }

          if tokens.count >= maxTokens {
            return .stop
          } else {
            return .more
          }
        }
      }

      // update the text if needed, e.g. we haven't displayed because of displayEveryNTokens
      if result.output != self.output {
        self.output = result.output
      }
      self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"

    } catch {
      output = "Failed: \(error)"
    }

    running = false
  }
}

public func mlxGenerate(
  input: LMInput, parameters: GenerateParameters, context: ModelContext,
  didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
  let iterator = try TokenIterator(
    input: input, model: context.model, parameters: parameters)
  return generate(
    input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}
