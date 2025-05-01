//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

import Foundation
import Hub

/// Represents a Hugging Face repository with its identity and authentication
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class HuggingFaceRepo: Codable, Equatable, Hashable {
    /// The default Hugging Face API endpoint
    public static let defaultDownloadBase = URL(string: "https://huggingface.co/api/")!
    
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
    public let downloadBase: URL
    
    /// The full repository identifier in the format "owner/repository"
    public var identifier: String {
        "\(owner)/\(repository)"
    }
    
    public init(owner: String = HuggingFaceRepo.defaultOwner, 
                repository: String = HuggingFaceRepo.defaultRepository, 
                token: String? = nil, 
                downloadBase: URL = HuggingFaceRepo.defaultDownloadBase) {
        self.owner = owner
        self.repository = repository
        self.token = token
        self.downloadBase = downloadBase
    }
    
    public init(_ identifier: String, token: String? = nil, downloadBase: URL = HuggingFaceRepo.defaultDownloadBase) {
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
    
    /// Downloads model files to a temporary location
    ///
    /// - Parameters:
    ///   - model: The model name to download
    ///   - useBackgroundSession: Whether to use a background download session
    ///   - progressCallback: Optional callback for download progress
    /// - Returns: URL to the temporary folder containing the downloaded model files
    public func downloadModelFiles(
        model: String,
        useBackgroundSession: Bool = false,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        let hubApi = HubApi(
            downloadBase: downloadBase,
            hfToken: token,
            useBackgroundSession: useBackgroundSession
        )
        
        // Download to temporary location
        let tempModelFolder = try await hubApi.snapshot(
            from: identifier,
            matching: ["*\(model)/*"]
        ) { progress in
            progressCallback?(progress)
        }
        
        return tempModelFolder
    }
} 