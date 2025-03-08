// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import Tokenizers

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
  _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (URL) throws -> M {
  { url in
    let configuration = try JSONDecoder().decode(
      C.self, from: Data(contentsOf: url))
    return modelInit(configuration)
  }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(hub:configuration:progressHandler:)``.
public class ModelTypeRegistry: @unchecked Sendable {

  // Note: using NSLock as we have very small (just dictionary get/set)
  // critical sections and expect no contention.  this allows the methods
  // to remain synchronous.
  private let lock = NSLock()

  private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
    "phi": create(PhiConfiguration.self, PhiModel.init),
    "phi3": create(Phi3Configuration.self, Phi3Model.init),
    "openelm": create(OpenElmConfiguration.self, OpenELMModel.init),
  ]

  /// Add a new model to the type registry.
  public func registerModelType(
    _ type: String, creator: @Sendable @escaping (URL) throws -> any LanguageModel
  ) {
    lock.withLock {
      creators[type] = creator
    }
  }

  /// Given a `modelType` and configuration file instantiate a new `LanguageModel`.
  public func createModel(configuration: URL, modelType: String) throws -> LanguageModel {
    let creator = lock.withLock {
      creators[modelType]
    }
    guard let creator else {
      throw ModelFactoryError.unsupportedModelType(modelType)
    }
    return try creator(configuration)
  }

}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class ModelRegistry: @unchecked Sendable {

  private let lock = NSLock()
  private var registry = Dictionary(uniqueKeysWithValues: all().map { ($0.name, $0) })

  static public let openelm270m4bit = MLXModelConfiguration(
    id: "mlx-community/OpenELM-270M-Instruct",
    // https://huggingface.co/apple/OpenELM
    defaultPrompt: "Once upon a time there was"
  )

  static public let phi35_mini_4bit = MLXModelConfiguration(
    id: "mlx-community/Phi-3.5-mini-instruct-4bit",
    defaultPrompt: "What is the gravity on Mars and the moon?",
    extraEOSTokens: ["<|end|>"]
  )

  static public let phi35_mini_8bit = MLXModelConfiguration(
    id: "HeadwatersJP/phi-3.5-mini-ft-mlx-8bit",
    defaultPrompt: "レポートを作成してください。",
    extraEOSTokens: ["<|end|>"]
  )

  static public let phi35_mini_8bit_lora = MLXModelConfiguration(
    id: "kokitakeishi/phi-3.5-lora-q8-mlx",
    defaultPrompt: "Microsoftが出しているサービス一覧",
    extraEOSTokens: ["<|end|>"]
  )

  static public let phi4_mini_4bit = MLXModelConfiguration(
    id: "lokinfey/Phi-4-mini-mlx-int4",
    defaultPrompt: "レポートを作成してください。",
    extraEOSTokens: ["<|end|>"]
  )

  static public let phi4_mini_8bit = MLXModelConfiguration(
    id: "kokitakeishi/Phi-4-mini-mlx-8bit",
    defaultPrompt: "Microsoftが出しているサービス一覧",
    extraEOSTokens: ["<|end|>"]
  )

  static public let phi4_2bit = MLXModelConfiguration(
    id: "mlx-community/phi-4-2bit",
    defaultPrompt: "List of Elon Musk's companies",
    extraEOSTokens: ["<|end|>"]
  )

  static public let phi4_4bit = MLXModelConfiguration(
    id: "mlx-community/phi-4-4bit",
    defaultPrompt: "レポートを作成してください。",
    extraEOSTokens: ["<|end|>"]
  )

  private static func all() -> [MLXModelConfiguration] {
    [
      openelm270m4bit,
      phi35_mini_4bit,
      phi35_mini_8bit,
      phi35_mini_8bit_lora,
      phi4_2bit,
      phi4_4bit,
      phi4_mini_4bit,
      phi4_mini_8bit,
    ]
  }

  public func register(configurations: [MLXModelConfiguration]) {
    lock.withLock {
      for c in configurations {
        registry[c.name] = c
      }
    }
  }

  public func configuration(id: String) -> MLXModelConfiguration {
    lock.withLock {
      if let c = registry[id] {
        return c
      } else {
        return MLXModelConfiguration(id: id)
      }
    }
  }
}

private struct LLMUserInputProcessor: UserInputProcessor {

  let tokenizer: Tokenizer
  let configuration: MLXModelConfiguration

  internal init(tokenizer: any Tokenizer, configuration: MLXModelConfiguration) {
    self.tokenizer = tokenizer
    self.configuration = configuration
  }

  func prepare(input: UserInput) throws -> LMInput {
    do {
      let messages = input.prompt.asMessages()
      let promptTokens = try tokenizer.applyChatTemplate(messages: messages)
      return LMInput(tokens: MLXArray(promptTokens))
    } catch {
      // #150 -- it might be a TokenizerError.chatTemplate("No chat template was specified")
      // but that is not public so just fall back to text
      let prompt = input.prompt
        .asMessages()
        .compactMap { $0["content"] }
        .joined(separator: ". ")
      let promptTokens = tokenizer.encode(text: prompt)
      return LMInput(tokens: MLXArray(promptTokens))
    }
  }
}

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let MLXModelContainer = try await LLMModelFactory.shared.loadContainer(
///     configuration: ModelRegistry.llama3_8B_4bit)
/// ```
public class LLMModelFactory: ModelFactory {

  public static let shared = LLMModelFactory()
  /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
  public let typeRegistry = ModelTypeRegistry()

  /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
  public let modelRegistry = ModelRegistry()

  public func configuration(id: String) -> MLXModelConfiguration {
    modelRegistry.configuration(id: id)
  }

  public func _load(
    hub: HubApi, configuration: MLXModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
  ) async throws -> ModelContext {
    // download weights and config
    let modelDirectory = try await downloadModel(
      hub: hub, configuration: configuration, progressHandler: progressHandler)
    print(modelDirectory)
    // load the generic config to unerstand which model and how to load the weights
    let configurationURL = modelDirectory.appending(component: "config.json")
    let baseConfig = try JSONDecoder().decode(
      BaseConfiguration.self, from: Data(contentsOf: configurationURL))
    let model = try typeRegistry.createModel(
      configuration: configurationURL, modelType: baseConfig.modelType)

    // apply the weights to the bare model
    try loadWeights(
      modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization)
    let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)
    return .init(
      configuration: configuration, model: model,
      processor: LLMUserInputProcessor(tokenizer: tokenizer, configuration: configuration),
      tokenizer: tokenizer)
  }

}
