//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

import Foundation

/// Represents the different sizes of Whisper models, without language or version specifics.
public enum ModelSize: String, CaseIterable, Comparable {
    case tiny, base, small, medium, large

    /// Provides an ordering for model sizes, smallest to largest.
    private var sortOrder: Int {
        switch self {
        case .tiny: return 0
        case .base: return 1
        case .small: return 2
        case .medium: return 3
        case .large: return 4
        }
    }

    /// Initializes a ModelSize from a model name string.
    /// - Parameter modelName: The name of the model (e.g., "openai_whisper-tiny.en", "openai_whisper-large")
    /// - Returns: The corresponding ModelSize, or nil if no size can be determined
    public init?(modelName: String) {
        for sizeCase in ModelSize.allCases.reversed() { // Check from largest to smallest to catch "large-v3" as "large"
            if modelName.lowercased().contains(sizeCase.rawValue) {
                self = sizeCase
                return
            }
        }
        return nil
    }

    public static func < (lhs: ModelSize, rhs: ModelSize) -> Bool {
        return lhs.sortOrder < rhs.sortOrder
    }
}

/// Represents a range of model sizes, with minimum and maximum bounds.
public struct ModelSizeRange {
    public let minimumSize: ModelSize
    public let maximumSize: ModelSize
    
    public init(minimumSize: ModelSize = .base, maximumSize: ModelSize = .large) {
        self.minimumSize = minimumSize
        self.maximumSize = maximumSize
    }
    
    /// Returns a new range that satisfies both this range and the other.
    public func combined(with other: ModelSizeRange) -> ModelSizeRange {
        let newMinimumSize = max(self.minimumSize, other.minimumSize)
        let newMaximumSize = min(self.maximumSize, other.maximumSize)
        return ModelSizeRange(minimumSize: newMinimumSize, maximumSize: newMaximumSize)
    }

    /// Checks if a given model size falls within this range.
    /// - Parameter size: The model size to check
    /// - Returns: True if the size is within the range (inclusive of bounds), false otherwise
    public func contains(_ size: ModelSize) -> Bool {
        return size >= minimumSize && size <= maximumSize
    }

    /// Checks if a model name's size falls within this range.
    /// - Parameter modelName: The name of the model to check
    /// - Returns: True if the model's size is within the range, false otherwise
    public func contains(modelName: String) -> Bool {
        guard let size = ModelSize(modelName: modelName) else {
            return false
        }
        return contains(size)
    }
}

/// Encapsulates constraints for model selection.
public struct ModelConstraint {
    public var sizeRange: ModelSizeRange
    public var isMultilingual: Bool

    public init(sizeRange: ModelSizeRange = ModelSizeRange(), isMultilingual: Bool = true) {
        self.sizeRange = sizeRange
        self.isMultilingual = isMultilingual
    }
    
    /// Convenience initializer that takes minimumSize and maximumSize directly
    public init(minimumSize: ModelSize = .base, isMultilingual: Bool = true, maximumSize: ModelSize = .large) {
        self.sizeRange = ModelSizeRange(minimumSize: minimumSize, maximumSize: maximumSize)
        self.isMultilingual = isMultilingual
    }

    /// Returns a new constraint that satisfies both this constraint and the other.
    public func combined(with other: ModelConstraint) -> ModelConstraint {
        return ModelConstraint(
            sizeRange: self.sizeRange.combined(with: other.sizeRange),
            isMultilingual: self.isMultilingual || other.isMultilingual
        )
    }

    /// Returns a new constraint, ensuring it is multilingual.
    public var asMultilingual: ModelConstraint {
        return ModelConstraint(sizeRange: self.sizeRange, isMultilingual: true)
    }
}

/// Represents the state of the remote model support configuration loading process.
internal enum RemoteConfigState: Equatable {
    /// The remote configuration is currently being fetched.
    case loading
    /// The remote configuration was successfully loaded.
    case loaded
    /// Fetching the remote configuration failed with an error.
    case failed(Error)

    public static func == (lhs: RemoteConfigState, rhs: RemoteConfigState) -> Bool {
        switch (lhs, rhs) {
        case (.loading, .loading):
            return true
        case (.loaded, .loaded):
            return true
        case (.failed, .failed):
            return true
        default:
            return false
        }
    }
}

/// A local repository for managing Whisper models and coordinating with a Hugging Face repo.
/// It handles downloading, querying, and managing local models.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class ModelRepo {
    // MARK: - Properties
    
    /// The Hugging Face repository configuration
    public let huggingFaceRepo: HuggingFaceRepo
    
    /// The local directory where models are stored
    public let localDirectory: URL
    
    /// Whether to use background download sessions
    /// WARNING: Background downloads require additional app implementation.
    /// Your app must implement URLSession background task handling in AppDelegate and call
    /// `completeBackgroundDownload(for:)` when downloads finish. See Apple's documentation on
    /// background URLSession for details.
    public let useBackgroundDownloadSession: Bool
    
    /// Indicates whether the remote configuration has been loaded
    public private(set) var isRemoteConfigLoaded: Bool = false
    
    /// The detailed state of the remote configuration loading process (internal use)
    private var remoteConfigState: RemoteConfigState = .loading
    
    /// The configuration for model support - starts with fallback and updates when remote config downloads
    public private(set) var modelSupportConfig: ModelSupportConfig
    
    /// Task that handles the async loading of the remote configuration
    private var configLoadingTask: Task<Void, Never>?
    
    /// Returns the default local directory for storing models based on the repository identifier
    public static func defaultLocalDirectory(for repoIdentifier: String) -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent("huggingface/models/\(repoIdentifier)")
    }
    
    // MARK: - Initialization
    
    /// Initializes a new ModelRepo instance synchronously with fallback configuration
    /// and starts an async task to fetch the remote configuration.
    ///
    /// - Parameters:
    ///   - huggingFaceRepo: The Hugging Face repository configuration
    ///   - localDirectory: The local directory where models are stored
    ///   - useBackgroundDownloadSession: Whether to use background download sessions.
    ///     WARNING: This requires additional app implementation to handle background task completion.
    ///     If set to true, your app must implement URLSession background task handling in AppDelegate
    ///     and call `completeBackgroundDownload(for:)` when downloads finish.
    ///   - downloadRemoteConfig: Whether to download the remote configuration
    public init(
        huggingFaceRepo: HuggingFaceRepo = .init(),
        localDirectory: URL? = nil,
        useBackgroundDownloadSession: Bool = false,
        downloadRemoteConfig: Bool = true
    ) {
        self.huggingFaceRepo = huggingFaceRepo
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
        
        // Use provided directory or create default based on the repo
        self.localDirectory = localDirectory ?? ModelRepo.defaultLocalDirectory(for: huggingFaceRepo.identifier)
        
        // Start with fallback configuration
        self.modelSupportConfig = Constants.fallbackModelSupportConfig
        
        // Start async task to fetch remote configuration only if requested
        if downloadRemoteConfig {
            self.remoteConfigState = .loading // Start in loading state
            configLoadingTask = Task {
                do {
                    let remoteConfig = try await huggingFaceRepo.fetchModelSupportConfig()
                    self.modelSupportConfig = remoteConfig
                    self.isRemoteConfigLoaded = true
                    self.remoteConfigState = .loaded // Update state on success
                    Logging.debug("ModelRepo successfully loaded remote configuration")
                } catch {
                    self.remoteConfigState = .failed(error) // Update state on failure
                    Logging.error("Error fetching remote config: \(error). Using fallback configuration.")
                }
            }
        } else {
            // If not downloading remote, consider the fallback loaded immediately
            self.isRemoteConfigLoaded = true
            self.remoteConfigState = .loaded
        }
    }
    
    /// Waits for the asynchronous task fetching the remote model support configuration to complete.
    /// Call this after initializing `ModelRepo` if you need to ensure that the latest remote configuration
    /// has been loaded (or attempted) before proceeding. Methods like `recommendedModels` use the
    /// configuration available at the time they are called; without waiting, they might initially use
    /// the fallback configuration.
    public func waitForRemoteConfig() async {
        await configLoadingTask?.value
    }
    
    // MARK: - Device Information
    
    /// Returns the current device name
    public static func deviceName() -> String {
        #if !os(macOS) && !targetEnvironment(simulator)
        var utsname = utsname()
        uname(&utsname)
        return withUnsafePointer(to: &utsname.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) {
                String(cString: $0)
            }
        }
        #else
        return ProcessInfo.hwModel
        #endif
    }
    
    // MARK: - Model Support
    
    /// Returns the model support for the current device
    public func modelSupport() -> ModelSupport {
        modelSupportConfig.modelSupport(for: Self.deviceName())
    }
    
    /// Returns the model support for a specific device
    public func modelSupport(for deviceName: String) -> ModelSupport {
        modelSupportConfig.modelSupport(for: deviceName)
    }
    
    /// Returns the recommended models for a device
    /// 
    /// - Parameter deviceName: The device identifier to get model support for.
    ///   Pass nil to use the current device.
    /// - Returns: An array of model names ordered by preference, with the default model first
    ///   (if available) followed by models ordered from smallest to largest.
    public func recommendedModels(device deviceName: String? = nil) -> [String] {
        let support = deviceName != nil ? modelSupport(for: deviceName!) : modelSupport()
        return orderModelsByPreference(support: support)
    }
    
    /// Returns the recommended models for a specific language, taking into account
    /// language complexity, resource requirements, and multilingual preference.
    ///
    /// - Parameters:
    ///   - language: The language code to find models for.
    ///   - deviceName: The device identifier. Pass nil for the current device.
    ///   - constraint: The constraints to apply for model selection, like minimum size and multilingual preference.
    /// - Returns: An array of model names ordered by preference.
    public func recommendedModels(forLanguage language: String, device deviceName: String? = nil, constraint: ModelConstraint = ModelConstraint()) -> [String] {
        let englishOnly = ["en", "english"]
        let wellResourcedEuropean = ["es", "fr", "de", "it", "pt", "nl", "pl", "ro", "ca", "sv", "no", "da"]
        let mediumResourced = ["ru", "uk", "cs", "fi", "hu", "el", "tr", "bg", "sr", "hr", "sl"]
        let complexScriptOrTonal = ["zh", "ja", "ko", "ar", "hi", "th", "vi", "fa", "he", "ur"]
        
        let currentSupport = deviceName != nil ? modelSupport(for: deviceName!) : modelSupport()
        let availableModels = currentSupport.supported
        
        let inputLanguage = language.lowercased()
        let primaryLanguageCode: String
        
        if inputLanguage.contains("-") {
            primaryLanguageCode = inputLanguage.split(separator: "-").first?.lowercased() ?? inputLanguage
        } else {
            primaryLanguageCode = inputLanguage
        }
        
        var filteredModels: [String]
        let isEnglish = englishOnly.contains(primaryLanguageCode)

        if isEnglish {
            if !constraint.isMultilingual {
                let englishSpecificModels = availableModels.filter { $0.contains(".en") }
                if !englishSpecificModels.isEmpty {
                    filteredModels = englishSpecificModels
                    Logging.debug("Recommending English-specific models for 'en' due to isMultilingual:false preference.")
                } else {
                    filteredModels = availableModels.filter { !$0.contains(".") || $0.contains(".en") }
                    Logging.debug("No English-specific (.en) models found for 'en' with isMultilingual:false. Falling back to general English-compatible models.")
                }
            } else {
                filteredModels = availableModels.filter { !$0.contains(".") || $0.contains(".en") }
            }
        } else {
            if !constraint.isMultilingual {
                Logging.info("Multilingual model is required for language '\(primaryLanguageCode)'. Ignoring isMultilingual:false preference. Consider providing isMultilingual:true.")
            }
            filteredModels = availableModels.filter { !$0.contains(".en") }
        }
        
        if complexScriptOrTonal.contains(primaryLanguageCode) {
            filteredModels = filteredModels.filter {
                !$0.contains("tiny") && (!$0.contains("base") || $0.contains("large"))
            }
        } else if mediumResourced.contains(primaryLanguageCode) {
            filteredModels = filteredModels.filter {
                !$0.contains("tiny")
            }
        } else if !wellResourcedEuropean.contains(primaryLanguageCode) && !isEnglish {
            filteredModels = filteredModels.filter {
                !$0.contains("tiny") && (!$0.contains("base") || $0.contains("large"))
            }
        }
        
        if filteredModels.isEmpty {
            if isEnglish {
                filteredModels = availableModels.filter { !$0.contains(".") || $0.contains(".en") }
            } else {
                filteredModels = availableModels.filter { !$0.contains(".en") }
            }
            Logging.debug("Initial filtering led to empty list for lang '\(primaryLanguageCode)'. Reverted to broader language-appropriate list: \(filteredModels.count) models.")
        }
        
        let defaultModel = currentSupport.default
        let includeDefault = filteredModels.contains(defaultModel)
        
        return orderModelsByPreference(support: ModelSupport(
            default: includeDefault ? defaultModel : filteredModels.first ?? defaultModel,
            supported: filteredModels
        ))
    }
    
    /// Returns the recommended models for multiple languages, finding the smallest model
    /// that can adequately handle all specified languages.
    ///
    /// - Parameters:
    ///   - languages: Array of language codes. Accepts both simple ISO 639 codes
    ///     (e.g., "en", "zh", "pt") and BCP-47 locale-specific codes (e.g., "en-US", "zh-Hant", "pt-PT").
    ///   - deviceName: The device identifier to get model support for.
    ///     Pass nil to use the current device.
    ///   - constraint: The constraints to apply for model selection.
    /// - Returns: An array of model names ordered by preference, with the default model first
    ///   (if available) followed by models ordered from smallest to largest.
    public func recommendedModels(forLanguages languages: [String], device deviceName: String? = nil, constraint: ModelConstraint = ModelConstraint()) -> [String] {
        if languages.isEmpty {
            if constraint.isMultilingual {
                return recommendedModels(device: deviceName)
            } else {
                // If no languages are specified but isMultilingual is false, assume English-focused models are desired.
                return recommendedModels(forLanguage: "en", device: deviceName, constraint: ModelConstraint(isMultilingual: false))
            }
        }
        
        let isEnglishOnlyRequest = languages.allSatisfy { ["en", "english"].contains($0.lowercased()) }

        var iterationConstraint = constraint
        if !isEnglishOnlyRequest && !constraint.isMultilingual {
            Logging.info("Multilingual model is required when non-English languages are specified. Overriding isMultilingual:false.")
            iterationConstraint = constraint.asMultilingual
        }
        
        var allLanguageModels: [Set<String>] = []
        let effectiveConstraint = iterationConstraint

        for language in languages {
            // Pass the possibly adjusted `effectiveMultilingual` flag down via effectiveConstraint.
            let supportForLanguage = recommendedModels(forLanguage: language, device: deviceName, constraint: effectiveConstraint)
            allLanguageModels.append(Set(supportForLanguage))
        }
        
        // Find the intersection of supported models across all languages
        guard var intersection = allLanguageModels.first else {
            return recommendedModels(device: deviceName) // Fallback to device default if no languages
        }
        
        for models in allLanguageModels.dropFirst() {
            intersection = intersection.intersection(models)
        }
        
        // Convert back to array and maintain original ordering
        let currentSupport = recommendedModels(device: deviceName)
        let supportedModels = currentSupport.filter { intersection.contains($0) }
        
        // If no models support all languages, return the device default
        if supportedModels.isEmpty {
            return currentSupport
        }
        
        // Get the device support to determine the default model
        let deviceSupport = deviceName != nil ? modelSupport(for: deviceName!) : modelSupport()
        
        // Order the filtered models by preference
        return orderModelsByPreference(support: ModelSupport(
            default: deviceSupport.default,
            supported: supportedModels
        ))
    }
    
    /// Orders models by preference, with the default model first (if available)
    /// followed by models ordered from smallest to largest.
    private func orderModelsByPreference(support: ModelSupport) -> [String] {
        var orderedModels = [String]()
        
        // Start with the default model if it's in the supported list
        if support.supported.contains(support.default) {
            orderedModels.append(support.default)
        }
        
        // Sort remaining models by size
        let sizeOrder = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v3", "large"]
        let remainingModels = support.supported.filter { $0 != support.default }
        
        let sortedModels = remainingModels.sorted { firstModel, secondModel in
            // Extract the base size without any additional qualifiers
            let firstModelBase = sizeOrder.first(where: { firstModel.contains($0) }) ?? firstModel
            let secondModelBase = sizeOrder.first(where: { secondModel.contains($0) }) ?? secondModel
            
            let firstIndex = sizeOrder.firstIndex(where: { firstModelBase.contains($0) }) ?? sizeOrder.count
            let secondIndex = sizeOrder.firstIndex(where: { secondModelBase.contains($0) }) ?? sizeOrder.count
            
            if firstIndex == secondIndex {
                // If same size, sort alphabetically
                return firstModel < secondModel
            }
            
            return firstIndex < secondIndex
        }
        
        orderedModels.append(contentsOf: sortedModels)
        return orderedModels
    }
    
    // MARK: - Local Models
    
    /// Returns the list of locally downloaded models
    public func localModels() throws -> [String] {
        guard FileManager.default.fileExists(atPath: localDirectory.path) else {
            return []
        }
        
        let downloadedModels = try FileManager.default.contentsOfDirectory(atPath: localDirectory.path)
        return Self.formatModelFiles(downloadedModels)
    }
    
    /// Returns the list of recommended models that are already downloaded
    public func recommendedModelsAvailableLocally() throws -> [String] {
        let local = try localModels()
        let recommended = recommendedModels()
        return recommended.filter { local.contains($0) }
    }
    
    /// Returns the list of recommended models for a language that are already downloaded
    public func recommendedModelsAvailableLocally(forLanguage language: String, device deviceName: String? = nil, constraint: ModelConstraint = ModelConstraint()) throws -> [String] {
        let local = try localModels()
        let recommended = recommendedModels(forLanguage: language, device: deviceName, constraint: constraint)
        return recommended.filter { local.contains($0) }
    }
    
    /// Returns the list of recommended models that support all specified languages and are already downloaded
    ///
    /// - Parameters:
    ///   - languages: Array of language codes to find models for. Models must support ALL languages.
    ///   - deviceName: The device identifier to get model support for. Pass nil to use the current device.
    ///   - constraint: The constraints to apply for model selection.
    /// - Returns: Array of downloaded models that support all the specified languages
    public func recommendedModelsAvailableLocally(forLanguages languages: [String], device deviceName: String? = nil, constraint: ModelConstraint = ModelConstraint()) throws -> [String] {
        let local = try localModels()
        let recommended = recommendedModels(forLanguages: languages, device: deviceName, constraint: constraint)
        return recommended.filter { local.contains($0) }
    }
    
    // MARK: - Model Download
    
    /// Downloads a model to the local directory.
    ///
    /// This method ensures atomicity: the model variant is first downloaded completely to a 
    /// temporary location, and only then moved to its final destination within the repository. 
    /// This prevents a corrupted or incomplete model in case of interruptions.
    ///
    /// - Parameters:
    ///   - model: The model name to download
    ///   - progressCallback: Optional callback to track download progress
    /// - Returns: The final URL to the model folder in the repository
    /// - Throws: An error if the model name can't be determined or if file operations fail
    public func download(
        model: String,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        // Create a unique temporary directory for this download operation
        let tempDownloadSessionDir = FileManager.default.temporaryDirectory.appendingPathComponent("ModelRepoDownload-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDownloadSessionDir, withIntermediateDirectories: true)
        
        defer {
            // Clean up the temporary download session directory
            try? FileManager.default.removeItem(at: tempDownloadSessionDir)
        }

        // Instruct HuggingFaceRepo to download the model files using our session-specific temporary directory as the base.
        // HuggingFaceRepo.downloadModelFiles (using HubApi) will download into a structure like:
        // tempDownloadSessionDir/repo.type/repo.id/modelName/
        // It will return the path: tempDownloadSessionDir/repo.type/repo.id/modelName/
        let downloadedModelVariantInTempDir = try await huggingFaceRepo.downloadModelFiles(
            model: model, 
            useBackgroundSession: useBackgroundDownloadSession,
            progressCallback: progressCallback,
            downloadToBase: tempDownloadSessionDir 
        )
        
        // Define the final destination for the model variant within the ModelRepo's localDirectory
        let finalModelVariantPath = localDirectory.appendingPathComponent(model)

        // Ensure the parent directory for the final model path exists (e.g., .../localDirectory/)
        try FileManager.default.createDirectory(at: localDirectory, withIntermediateDirectories: true)

        // Atomically move the downloaded model variant from the temporary location to the final destination
        // First, remove any existing model at the final destination if it exists (to allow overwrite by move)
        if FileManager.default.fileExists(atPath: finalModelVariantPath.path) {
            try FileManager.default.removeItem(at: finalModelVariantPath)
        }
        try FileManager.default.moveItem(at: downloadedModelVariantInTempDir, to: finalModelVariantPath)

        Logging.debug("Model '\(model)' downloaded and moved to '\(finalModelVariantPath.path)'")
        return finalModelVariantPath
    }
    
    /// Seeds a Hugging Face model from a local file URL into the repository
    ///
    /// This method is specifically for seeding models into the Hugging Face model directory
    /// structure. It will place the model in the standard location: huggingface/models/argmaxinc/whisperkit-coreml
    /// 
    /// This method is useful when you ship a model with your app and want to make it available.
    /// With a model shipped in the app bundle, you are guaranteed to have the model available
    /// when the app is installed.
    ///
    /// - Parameters:
    ///   - sourceURL: The file URL pointing to the directory containing the model files to import.
    ///                The directory itself will be copied.
    ///   - overwriteExisting: If true (default), any existing model with the same name will be
    ///                        deleted before importing. If false, and a model with the same name
    ///                        already exists, the function will return the URL of the existing
    ///                        model without performing the import.
    /// - Returns: The final URL of the imported model directory within the repository.
    /// - Throws: An error if file operations (directory creation, removal, copy) fail.
    @discardableResult
    public func seedHuggingFaceModel(
        from sourceURL: URL,
        overwriteExisting: Bool = true
    ) throws -> URL {
        let modelConfigURL = sourceURL.appendingPathComponent("config.json")
        var modelNameToUse: String

        do {
            let configData = try Data(contentsOf: modelConfigURL)
            if let json = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
               let nameOrPath = json["_name_or_path"] as? String {
                // Transform "owner/variant" to "owner_variant" for the subdirectory name
                modelNameToUse = nameOrPath.replacingOccurrences(of: "/", with: "_")
                Logging.debug("Derived model name '\(modelNameToUse)' from config.json (_name_or_path: \(nameOrPath))")
            } else {
                modelNameToUse = sourceURL.lastPathComponent
                Logging.debug("Could not find '_name_or_path' in config.json or config.json not valid JSON. Using sourceURL.lastPathComponent: '\(modelNameToUse)'")
            }
        } catch {
            modelNameToUse = sourceURL.lastPathComponent
            Logging.debug("Could not read config.json from sourceURL. Using sourceURL.lastPathComponent: '\(modelNameToUse)'. Error: \(error)")
        }

        // Create the final destination path in the standard Hugging Face location
        let finalModelFolder = localDirectory.appendingPathComponent(modelNameToUse)

        // Check if model already exists and if we should overwrite
        let modelExists = FileManager.default.fileExists(atPath: finalModelFolder.path)

        if modelExists && !overwriteExisting {
            // Model exists and we shouldn't overwrite, return existing path
            Logging.debug("Model '\\(modelNameToUse)' already exists and overwriteExisting is false. Skipping import.")
            return finalModelFolder
        }

        // Ensure the parent repository directory exists
        try FileManager.default.createDirectory(
            at: localDirectory,
            withIntermediateDirectories: true
        )

        // Remove the existing model folder if it exists (and we are allowed to overwrite or it didn't exist before)
        if modelExists {
            try FileManager.default.removeItem(at: finalModelFolder)
        }

        // Copy the item from source URL to the final location
        try FileManager.default.copyItem(at: sourceURL, to: finalModelFolder)

        Logging.debug("Successfully seeded model \\(modelNameToUse) from \\(sourceURL.path) to \\(finalModelFolder.path)")

        return finalModelFolder
    }
    
    /// Deletes a downloaded model
    public func delete(model: String) throws {
        let modelFolder = localDirectory.appendingPathComponent(model)
        try FileManager.default.removeItem(at: modelFolder)
    }
    
    /// Deletes all downloaded models from the local directory
    /// - Returns: Array of model names that were successfully deleted
    @discardableResult
    public func deleteAllDownloadedModels() throws -> [String] {
        // Get all locally downloaded models
        let models = try localModels()
        var deletedModels: [String] = []
        
        // Track any errors that occur during deletion
        var deletionError: Error?
        
        // Try to delete each model
        for model in models {
            do {
                try delete(model: model)
                deletedModels.append(model)
            } catch {
                // If an error occurs, remember it but continue trying to delete others
                deletionError = error
                Logging.error("Failed to delete model \(model): \(error)")
            }
        }
        
        // If we encountered any errors but deleted some models, throw the error after recording deletions
        if let error = deletionError, !deletedModels.isEmpty {
            throw error
        }
        
        return deletedModels
    }
    
    /// Completes a background download by moving the model from a temporary location to the final model repository
    /// 
    /// Call this method from your app's URLSessionDownloadDelegate when a background download completes.
    /// This is required for apps that use background downloads with `useBackgroundDownloadSession` set to true.
    ///
    /// Example implementation in your AppDelegate:
    /// ```swift
    /// func urlSession(_ session: URLSession, 
    ///                 downloadTask: URLSessionDownloadTask, 
    ///                 didFinishDownloadingTo location: URL) {
    ///     guard let originalURL = downloadTask.originalRequest?.url else {
    ///         print("Error: Could not get original request URL")
    ///         return
    ///     }
    ///     
    ///     do {
    ///         // Complete the download by passing both the temporary location and original URL
    ///         let finalLocation = try modelRepo.completeBackgroundDownload(
    ///             tempURL: location,
    ///             originalRequestURL: originalURL
    ///         )
    ///         
    ///         print("Background download completed and moved to: \(finalLocation)")
    ///     } catch {
    ///         print("Failed to complete background download: \(error)")
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - tempURL: The temporary URL where the downloaded files are stored
    ///   - originalRequestURL: The original URL that was requested. This contains the model name
    ///           in the path and is used to identify the downloaded model.
    /// - Returns: The final URL to the model folder in the repository
    /// - Throws: An error if the model name can't be determined or if file operations fail
    public func completeBackgroundDownload(
        tempURL: URL,
        originalRequestURL: URL
    ) throws -> URL {
        // Extract model name from original request URL
        let pathComponents = originalRequestURL.pathComponents
        
        // Check for standard model names in the path
        let modelKeywords = ["tiny", "base", "small", "medium", "large"]
        
        // Try to identify the model name from the URL path
        let modelName: String
        
        // Find the first path component that contains any of the model keywords
        if let modelComponent = pathComponents.first(where: { component in 
            modelKeywords.contains { component.contains($0) }
        }) {
            modelName = modelComponent
        } else {
            // Fallback: Use last path component
            modelName = pathComponents.last ?? "unknown-model"
        }
        
        // Create the final destination path
        let finalModelFolder = localDirectory.appendingPathComponent(modelName)
        
        // Ensure the parent directory exists
        try FileManager.default.createDirectory(
            at: localDirectory,
            withIntermediateDirectories: true
        )
        
        // Remove the existing model folder if it exists
        if FileManager.default.fileExists(atPath: finalModelFolder.path) {
            try FileManager.default.removeItem(at: finalModelFolder)
        }
        
        // Move the downloaded files to the final location
        try FileManager.default.moveItem(at: tempURL, to: finalModelFolder)
        
        return finalModelFolder
    }
    
    // MARK: - Convenience Download Methods
    
    /// Extracts the base ModelSize from a model name string.
    private func getModelSizeFromName(_ modelName: String) -> ModelSize? {
        return ModelSize(modelName: modelName)
    }

    /// Checks if a model name meets or exceeds a minimum size requirement.
    private func modelNameSatisfiesMinimumSize(_ modelName: String, minSize: ModelSize) -> Bool {
        guard let modelSize = getModelSizeFromName(modelName) else {
            return false // Cannot determine size from name
        }
        return modelSize >= minSize
    }

    /// Checks if a model name is within a maximum size requirement.
    private func modelNameSatisfiesMaximumSize(_ modelName: String, maxSize: ModelSize) -> Bool {
        guard let modelSize = getModelSizeFromName(modelName) else {
            return true // Cannot determine size from name, so can't enforce max; effectively passes
        }
        return modelSize <= maxSize
    }

    /// Determines the target model name based on recommended/downloaded lists and other criteria.
    private func determineTargetModel(
        recommended: [String],
        downloaded: [String],
        constraint: ModelConstraint,
        defaultModel: String
    ) -> String {
        var candidates = recommended

        // 1. Filter by multilingual requirement
        if constraint.isMultilingual {
            candidates = candidates.filter { !$0.contains(".en") }
        }
        // If not multilingualRequired, we accept both .en and multilingual models from the `recommended` list.
        // The `recommendedModels(forLanguage:"en")` method should provide a list suitable for English.

        // 2. Filter by size range
        candidates = candidates.filter { constraint.sizeRange.contains(modelName: $0) }

        // 3. If no candidates meet criteria, fall back to defaultModel immediately.
        //    (Ideally, defaultModel should also be checked, but current logic is to provide it as a last resort)
        if candidates.isEmpty {
            Logging.debug("No models in the recommended list satisfy the multilingual and size range criteria. Falling back to default model: \(defaultModel)")
            return defaultModel
        }

        // 4. Prioritize downloaded models that meet criteria
        let downloadedCandidates = candidates.filter { downloaded.contains($0) }
        
        let modelsToConsiderForOrdering: [String]
        if !downloadedCandidates.isEmpty {
            modelsToConsiderForOrdering = downloadedCandidates
            Logging.debug("Found downloaded models satisfying criteria: \(modelsToConsiderForOrdering)")
        } else {
            modelsToConsiderForOrdering = candidates
            Logging.debug("No downloaded models satisfy criteria. Considering recommended models for download: \(modelsToConsiderForOrdering)")
        }
        
        // 5. Order the chosen set of models and pick the best one.
        // The `defaultModel` passed here is a hint for ordering if it's part of modelsToConsiderForOrdering.
        // If not, the first of the sorted list will be a good primary candidate.
        let effectiveDefaultForOrdering = modelsToConsiderForOrdering.contains(defaultModel) ? defaultModel : modelsToConsiderForOrdering.first ?? defaultModel
        
        let orderedModels = orderModelsByPreference(support: ModelSupport(
            default: effectiveDefaultForOrdering, 
            supported: modelsToConsiderForOrdering
        ))

        if let bestChoice = orderedModels.first {
            Logging.debug("Best choice after ordering: \(bestChoice)")
            return bestChoice
        } else {
            // This case should ideally not be reached if candidates was not empty.
            // But as a final fallback if ordering somehow results in an empty list.
            Logging.debug("Ordering resulted in no best choice. Falling back to default model: \(defaultModel)")
            return defaultModel
        }
    }

    /// Downloads the best model for the specified languages if needed and returns its name.
    /// - Parameters:
    ///   - languages: Array of language codes to support
    ///   - constraint: Model constraints for size and multilingual capability. Defaults to .base, multilingual.
    ///   - progressCallback: Optional callback to track download progress
    /// - Returns: The name of the downloaded or existing model
    public func downloadedModel(
        forLanguages languages: [String],
        constraint: ModelConstraint = ModelConstraint(),
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> String {
        let recommendedForLang = recommendedModels(forLanguages: languages, constraint: constraint)
        let downloadedForLang = try recommendedModelsAvailableLocally(forLanguages: languages, constraint: constraint)
        let deviceDefaultModel = modelSupport().default

        let targetModel = determineTargetModel(
            recommended: recommendedForLang,
            downloaded: downloadedForLang,
            constraint: constraint, 
            defaultModel: deviceDefaultModel
        )

        let allDownloaded = try localModels()
        if allDownloaded.contains(targetModel) {
            return targetModel
        } else {
            _ = try await download(model: targetModel, progressCallback: progressCallback)
            return targetModel
        }
    }

    /// Downloads the best model for the current device if needed and returns its name.
    /// - Parameters:
    ///   - constraint: Model constraints for size and multilingual capability. Defaults to .base, multilingual.
    ///   - progressCallback: Optional callback to track download progress
    /// - Returns: The name of the downloaded or existing model
    public func downloadedModel(
        constraint: ModelConstraint = ModelConstraint(),
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> String {
        let recommendedOverall: [String]
        let downloadedOverall: [String]
        let deviceDefaultModel = modelSupport().default
        // Preserve maximumSize from original constraint when creating englishFocusedConstraint
        let englishFocusedConstraint = ModelConstraint(sizeRange: ModelSizeRange(minimumSize: constraint.sizeRange.minimumSize, maximumSize: constraint.sizeRange.maximumSize), isMultilingual: false)

        if !constraint.isMultilingual { // User specifically wants English-focused
            recommendedOverall = recommendedModels(forLanguage: "en", constraint: englishFocusedConstraint)
            downloadedOverall = try recommendedModelsAvailableLocally(forLanguage: "en", constraint: englishFocusedConstraint)
        } else {
            recommendedOverall = recommendedModels() // General recommendations for the device
            downloadedOverall = try recommendedModelsAvailableLocally()
        }
        
        let targetModel = determineTargetModel(
            recommended: recommendedOverall,
            downloaded: downloadedOverall,
            constraint: constraint, 
            defaultModel: deviceDefaultModel
        )

        if downloadedOverall.contains(targetModel) { // Check against the context-specific downloaded list
            // It's possible targetModel is from the broader `recommendedOverall` but not in context-specific `downloadedOverall`
            // A final check against all local models is safer.
            let allLocal = try localModels()
            if allLocal.contains(targetModel) {
                 return targetModel
            }
        }
        // If not in context-specific downloaded list, or not in allLocal, then download.
        _ = try await download(model: targetModel, progressCallback: progressCallback)
        return targetModel
    }

    // MARK: - Model Formats

    /// Formats raw model file paths into standardized model identifiers
    ///
    /// This method takes an array of raw file or directory paths and:
    /// 1. Filters them to include only valid model variants
    /// 2. Standardizes the format for model identifiers
    /// 3. Sorts them in a logical order by size/capability
    ///
    /// - Parameter modelFiles: The raw file or directory paths to format
    /// - Returns: An array of formatted model identifiers
    public static func formatModelFiles(_ modelFiles: [String]) -> [String] {
        // Extract model names from paths
        let modelNames = modelFiles.map { path -> String in
            let components = path.components(separatedBy: "/")
            return components[0]
        }
        
        // Sort models by size and name
        let sizeOrder = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v3", "large"]
        
        // Filter to only include valid model variants and sort them
        let validModels = modelNames.filter { name in
            sizeOrder.contains(where: { name.contains($0) })
        }.sorted { firstModel, secondModel in
            // Extract the base size without any additional qualifiers
            let firstModelBase = sizeOrder.first(where: { firstModel.contains($0) }) ?? firstModel
            let secondModelBase = sizeOrder.first(where: { secondModel.contains($0) }) ?? secondModel
            
            let firstIndex = sizeOrder.firstIndex(where: { firstModelBase.contains($0) }) ?? sizeOrder.count
            let secondIndex = sizeOrder.firstIndex(where: { secondModelBase.contains($0) }) ?? sizeOrder.count
            
            if firstIndex == secondIndex {
                // If same size, sort alphabetically
                return firstModel < secondModel
            }
            
            return firstIndex < secondIndex
        }
        
        // Remove duplicates while preserving order
        return Array(NSOrderedSet(array: validModels)) as! [String]
    }
    
    // MARK: - Testing Support
    
    /// Factory method to create a ModelRepo with a specific configuration - for testing only
    /// This allows tests to create a ModelRepo with a predefined configuration without network calls
    #if DEBUG
    internal static func forTesting(
        huggingFaceRepo: HuggingFaceRepo = .init(),
        localDirectory: URL? = nil,
        useBackgroundDownloadSession: Bool = false,
        modelSupportConfig: ModelSupportConfig = Constants.fallbackModelSupportConfig,
        initialRemoteConfigState: RemoteConfigState = .loaded // Default to loaded for tests
    ) -> ModelRepo {
        // Initialize without starting the remote download task
        let repo = ModelRepo(
            huggingFaceRepo: huggingFaceRepo,
            localDirectory: localDirectory,
            useBackgroundDownloadSession: useBackgroundDownloadSession,
            downloadRemoteConfig: false // Explicitly disable remote config loading
        )
        
        // Manually set the desired state for testing
        repo.modelSupportConfig = modelSupportConfig
        repo.remoteConfigState = initialRemoteConfigState
        // Determine if a config is considered "loaded" (either successfully or failed)
        repo.isRemoteConfigLoaded = (initialRemoteConfigState != .loading)
        
        // No need to manage tasks here anymore
        
        return repo
    }
    #endif
} 
