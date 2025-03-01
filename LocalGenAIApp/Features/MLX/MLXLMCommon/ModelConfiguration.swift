// Copyright © 2024 Apple Inc.

import Foundation
import Hub

/// Configuration for a given model name with overrides for prompts and tokens.
///
/// See e.g. `MLXLM.ModelRegistry` for an example of use.
public struct MLXModelConfiguration: Sendable {

    public enum Identifier: Sendable {
        case id(String)
        case directory(URL)
    }

    public var id: Identifier

    public var name: String {
        switch id {
        case .id(let string):
            string
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// pull the tokenizer from an alternate id
    public let tokenizerId: String?

    /// overrides for TokenizerModel/knownTokenizers -- useful before swift-transformers is updated
    public let overrideTokenizer: String?

    /// A reasonable default prompt for the model
    public let defaultPrompt: String

    /// Additional tokens to use for end of string
    public let extraEOSTokens: Set<String>

    public init(
        id: String, tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = [],
        preparePrompt: (@Sendable (String) -> String)? = nil
    ) {
        self.id = .id(id)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
    }

    public init(
        directory: URL, tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = []
    ) {
        
        let fileManager = FileManager.default
        let documentDirectoryFileURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let directoryPath = documentDirectoryFileURL.appendingPathComponent("mlx_model", isDirectory: true)
        do {
            try fileManager.createDirectory(at: directoryPath, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print("失敗した")
        }
        var pathString = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
       //パスを用意する
       pathString = "file://" + pathString + "/" + "mlx_model/"
       let path = URL(string: pathString)
       //保存するファイルを用意する
       let fileName = "config.json"
       let textPath = path!.appendingPathComponent(fileName)
        
        self.id = .directory(directory)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
    }

    public func modelDirectory(hub: HubApi = HubApi()) -> URL {
        switch id {
        case .id(let id):
            // download the model weights and config
            let repo = Hub.Repo(id: id)
            return hub.localRepoLocation(repo)

        case .directory(let directory):
            return directory
        }
    }
}
