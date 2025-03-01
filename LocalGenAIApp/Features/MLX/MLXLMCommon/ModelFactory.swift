// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

public enum ModelFactoryError: Error {
    case unsupportedModelType(String)
    case unsupportedProcessorType(String)
}

/// Context of types that work together to provide a ``LanguageModel``.
///
/// A ``ModelContext`` is created by ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This contains the following:
///
/// - ``MLXModelConfiguration`` -- identifier for the model
/// - ``LanguageModel`` -- the model itself, see ``generate(input:parameters:context:didGenerate:)``
/// - ``UserInputProcessor`` -- can convert ``UserInput`` into ``LMInput``
/// - `Tokenizer` -- the tokenizer used by ``UserInputProcessor``
///
/// See also ``ModelFactory/loadContainer(hub:configuration:progressHandler:)`` and
/// ``MLXModelContainer``.
public struct ModelContext {
    public let configuration: MLXModelConfiguration
    public let model: any LanguageModel
    public let processor: any UserInputProcessor
    public let tokenizer: Tokenizer

    public init(
        configuration: MLXModelConfiguration, model: any LanguageModel,
        processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) {
        self.configuration = configuration
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    }
}

public protocol ModelFactory: Sendable {

    /// Resolve a model identifier, e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit", into
    /// a ``MLXModelConfiguration``.
    ///
    /// This will either create a new (mostly unconfigured) ``MLXModelConfiguration`` or
    /// return a registered instance that matches the id.
    func configuration(id: String) -> MLXModelConfiguration

    func _load(
        hub: HubApi, configuration: MLXModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext

    func _loadContainer(
        hub: HubApi, configuration: MLXModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> MLXModelContainer
}

extension ModelFactory {

    /// Load a model identified by a ``MLXModelConfiguration`` and produce a ``ModelContext``.
    ///
    /// This method returns a ``ModelContext``.  See also
    /// ``loadContainer(hub:configuration:progressHandler:)`` for a method that
    /// returns a ``MLXModelContainer``.
    public func load(
        hub: HubApi = HubApi(hfToken: Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String), configuration: MLXModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContext {
        try await _load(hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    /// Load a model identified by a ``MLXModelConfiguration`` and produce a ``MLXModelContainer``.
    public func loadContainer(
        hub: HubApi = HubApi(hfToken: Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String), configuration: MLXModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXModelContainer {
        try await _loadContainer(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    public func _loadContainer(
        hub: HubApi = HubApi(hfToken: Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String), configuration: MLXModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXModelContainer {
        let context = try await _load(
            hub: hub, configuration: configuration, progressHandler: progressHandler)
        return MLXModelContainer(context: context)
    }

}
