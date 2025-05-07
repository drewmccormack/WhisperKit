//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

import Foundation
import Hub

/// Represents a Hugging Face repository with its identity and authentication
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class HuggingFaceRepo: Codable, Equatable, Hashable {
    /// The default repository owner
    public static let defaultOwner = "argmaxinc"
    
    /// The default repository name
    public static let defaultRepository = "whisperkit-coreml"
    
    /// The owner of the repository
    public let owner: String
    
    /// The name of the repository
    public let repository: String
    
    /// The authentication token for the repository. Can be nil for public repos
    public var token: String?
    
    /// The base URL for downloading from the repository. If nil, HubApi will use its default.
    public let downloadBase: URL?
    
    /// The full repository identifier in the format "owner/repository"
    public var identifier: String {
        "\(owner)/\(repository)"
    }
    
    public init(owner: String = HuggingFaceRepo.defaultOwner, 
                repository: String = HuggingFaceRepo.defaultRepository, 
                token: String? = nil, 
                downloadBase: URL? = nil) {
        self.owner = owner
        self.repository = repository
        self.token = token
        self.downloadBase = downloadBase
    }
    
    public init(_ identifier: String, token: String? = nil, downloadBase: URL? = nil) {
        let components = identifier.components(separatedBy: "/")
        guard components.count == 2 else {
            fatalError("Invalid Hugging Face repository identifier: \(identifier). Expected format: owner/repository")
        }
        self.owner = components[0]
        self.repository = components[1]
        self.token = token
        self.downloadBase = downloadBase
    }
    
    // MARK: - Hashable & Equatable
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(owner)
        hasher.combine(repository)
        hasher.combine(downloadBase)
    }
    
    public static func == (lhs: HuggingFaceRepo, rhs: HuggingFaceRepo) -> Bool {
        return lhs.owner == rhs.owner &&
               lhs.repository == rhs.repository &&
               lhs.downloadBase == rhs.downloadBase
    }
    
    // MARK: - API Methods
    
    /// Fetches the model support configuration from the repository.
    /// This method always returns a ModelSupportConfig, using a fallback if the fetch fails.
    public func fetchModelSupportConfig() async throws -> ModelSupportConfig {
        let hubApi = HubApi(downloadBase: downloadBase, hfToken: token)
        var modelSupportConfig = Constants.fallbackModelSupportConfig
        
        do {
            // HubApi.snapshot downloads the config.json to its cache and returns the repo root in that cache.
            let downloadedRepoRootInCache = try await hubApi.snapshot(from: identifier, matching: "config*")
            let decoder = JSONDecoder()
            let jsonData = try Data(contentsOf: downloadedRepoRootInCache.appendingPathComponent("config.json"))
            modelSupportConfig = try decoder.decode(ModelSupportConfig.self, from: jsonData)
        } catch {
            Logging.error("Error fetching model support config, using fallback: \(error)")
        }
        
        return modelSupportConfig
    }
    
    /// Downloads all files for a given model variant (matching `*modelName/*`) into a specified base location.
    /// 
    /// The `HubApi` used internally downloads files directly into a structured path within the `effectiveApiDownloadBase`.
    /// For example, if `effectiveApiDownloadBase` is `~/Temp/Downloads`, `repo.type` is `.models`,
    /// `repo.id` is `owner/repoName`, and `model` is `modelVariant`, files will be placed in
    /// `~/Temp/Downloads/models/owner/repoName/modelVariant/`.
    /// 
    /// This method returns the path to the `modelVariant` folder within the `effectiveApiDownloadBase` structure.
    /// If `effectiveApiDownloadBase` is intended as a temporary location (e.g., by `ModelRepo`), the caller is responsible
    /// for moving the returned model variant folder to a final persistent repository.
    ///
    /// - Parameters:
    ///   - model: The model variant name (e.g., "openai_whisper-base") used to construct the glob pattern `*modelName/*` for HubApi.
    ///   - useBackgroundSession: Whether to use a background download session.
    ///   - progressCallback: Optional callback for download progress.
    ///   - downloadToBase: Optional URL to use as the root for `HubApi` downloads. If nil, `self.downloadBase` is used.
    ///                     If both are nil, `HubApi` uses its own default (typically `Documents/huggingface`).
    /// - Returns: URL to the folder containing the downloaded model variant files within the `effectiveApiDownloadBase` structure.
    public func downloadModelFiles(
        model: String,
        useBackgroundSession: Bool = false,
        progressCallback: ((Progress) -> Void)? = nil,
        downloadToBase: URL? = nil
    ) async throws -> URL {
        let effectiveApiDownloadBase = downloadToBase ?? self.downloadBase

        let hubApi = HubApi(
            downloadBase: effectiveApiDownloadBase,
            hfToken: self.token,
            endpoint: "https://huggingface.co", 
            useBackgroundSession: useBackgroundSession
        )

        let repo = Hub.Repo(id: identifier)
        
        // HubApi.snapshot downloads files matching "*\(model)/*" 
        // into: effectiveApiDownloadBase/repo.type/repo.id/ (then files go into model-specific subdirs)
        // It returns the path: effectiveApiDownloadBase/repo.type/repo.id/
        let downloadedRepoRoot = try await hubApi.snapshot(from: repo, matching: ["*\(model)/*"]) { progress in
            progressCallback?(progress)
        }
        
        // The actual model variant files are in a subdirectory named `model` (the variant name)
        // within this `downloadedRepoRoot`.
        let modelVariantPathInEffectiveBase = downloadedRepoRoot.appendingPathComponent(model)

        var isDirectory: ObjCBool = false
        if !FileManager.default.fileExists(atPath: modelVariantPathInEffectiveBase.path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            let baseDesc = effectiveApiDownloadBase?.path ?? "HubApi default (Documents/huggingface)"
            throw WhisperError.modelsUnavailable("Model variant '\(model)' not found at expected path '\(modelVariantPathInEffectiveBase.path)' after download attempt using base '\(baseDesc)'.")
        }
        
        return modelVariantPathInEffectiveBase
    }
} 