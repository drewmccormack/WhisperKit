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
    
    /// The base URL for downloading from the repository
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
    
    /// Fetches the model support configuration from the repository
    /// This method always returns a ModelSupportConfig, using a fallback if the fetch fails
    public func fetchModelSupportConfig() async throws -> ModelSupportConfig {
        let hubApi = HubApi(downloadBase: downloadBase, hfToken: token)
        var modelSupportConfig = Constants.fallbackModelSupportConfig
        
        do {
            let configUrl = try await hubApi.snapshot(from: identifier, matching: "config*")
            let decoder = JSONDecoder()
            let jsonData = try Data(contentsOf: configUrl.appendingPathComponent("config.json"))
            modelSupportConfig = try decoder.decode(ModelSupportConfig.self, from: jsonData)
        } catch {
            Logging.error("Error fetching model support config, using fallback: \(error)")
        }
        
        return modelSupportConfig
    }
    
    /// Downloads model files, populating the specified base download location.
    /// 
    /// The `HubApi` used internally downloads files directly into a structure within the 
    /// `effectiveApiDownloadBase`. If this base is a temporary location (as orchestrated by a caller like `ModelRepo`), 
    /// the caller is then responsible for atomically moving the resulting model variant folder 
    /// to its final persistent repository to ensure integrity.
    ///
    /// - Parameters:
    ///   - model: The model name to download (used to construct glob pattern)
    ///   - useBackgroundSession: Whether to use a background download session
    ///   - progressCallback: Optional callback for download progress
    ///   - downloadToBase: Optional URL to use as the base for HubApi downloads. If nil, uses self.downloadBase (which itself might be nil, causing HubApi to use its default).
    /// - Returns: URL to the temporary folder containing the downloaded model variant files.
    public func downloadModelFiles(
        model: String,
        useBackgroundSession: Bool = false,
        progressCallback: ((Progress) -> Void)? = nil,
        downloadToBase: URL? = nil
    ) async throws -> URL {
        // Determine the download base for HubApi for this specific operation.
        // If `downloadToBase` is provided, it overrides `self.downloadBase` for this call.
        // If both are nil, HubApi will use its own default (typically .../Documents/huggingface).
        let effectiveApiDownloadBase = downloadToBase ?? self.downloadBase

        let hubApi = HubApi(
            downloadBase: effectiveApiDownloadBase, // Use the determined base
            hfToken: self.token,
            endpoint: "https://huggingface.co", // Explicitly set endpoint
            useBackgroundSession: useBackgroundSession
        )

        let repo = Hub.Repo(id: identifier)
        
        // HubApi.snapshot will download files matching "*\(model)/*" 
        // into a structure like: effectiveApiDownloadBase/repo.type/repo.id/model/file.txt
        // It returns the path: effectiveApiDownloadBase/repo.type/repo.id
        let downloadedRepoRoot = try await hubApi.snapshot(from: repo, matching: ["*\(model)/*"]) { progress in
            progressCallback?(progress)
        }

        // The actual model variant files are inside a subdirectory named `model` (the variant name)
        // within this `downloadedRepoRoot`.
        let modelVariantPathInEffectiveBase = downloadedRepoRoot.appendingPathComponent(model)

        // Verify the model variant path exists after download
        var isDirectory: ObjCBool = false
        if !FileManager.default.fileExists(atPath: modelVariantPathInEffectiveBase.path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            let baseDesc = effectiveApiDownloadBase?.path ?? "HubApi default (Documents/huggingface)"
            throw WhisperError.modelsUnavailable("Model variant '\(model)' not found at expected path '\(modelVariantPathInEffectiveBase.path)' after download attempt using base '\(baseDesc)'.")
        }
        
        return modelVariantPathInEffectiveBase // Return the path to the actual model variant directory
    }
} 