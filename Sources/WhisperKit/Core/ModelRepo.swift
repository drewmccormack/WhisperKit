//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

import Foundation

/// Represents the state of the remote model support configuration loading process.
public enum RemoteConfigState: Equatable {
    /// The remote configuration is currently being fetched.
    case loading
    /// The remote configuration was successfully loaded.
    case loaded
    /// Fetching the remote configuration failed with an error.
    case failed(Error)

    // Custom Equatable conformance to handle the associated Error value
    public static func == (lhs: RemoteConfigState, rhs: RemoteConfigState) -> Bool {
        switch (lhs, rhs) {
        case (.loading, .loading):
            return true
        case (.loaded, .loaded):
            return true
        case (.failed, .failed):
            // Consider two .failed states equal regardless of the specific error
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
    
    /// The state of the remote configuration loading process
    public private(set) var remoteConfigState: RemoteConfigState = .loading
    
    /// The configuration for model support - starts with fallback and updates when remote config loads
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
    /// language complexity and resource requirements
    ///
    /// - Parameters:
    ///   - language: The language code to find models for. Accepts both simple ISO 639 codes
    ///     (e.g., "en", "zh", "pt") and BCP-47 locale-specific codes (e.g., "en-US", "zh-Hant", "pt-PT").
    ///     For locale-specific codes, only the primary language part is used for determining support.
    ///   - deviceName: The device identifier to get model support for.
    ///     Pass nil to use the current device.
    ///
    /// - Returns: An array of model names ordered by preference, with the default model first
    ///   (if available) followed by models ordered from smallest to largest.
    public func recommendedModels(forLanguage language: String, device deviceName: String? = nil) -> [String] {
        // Language categories by resource requirements and script complexity
        let englishOnly = ["en", "english"]
        let wellResourcedEuropean = ["es", "fr", "de", "it", "pt", "nl", "pl", "ro", "ca", "sv", "no", "da"]
        let mediumResourced = ["ru", "uk", "cs", "fi", "hu", "el", "tr", "bg", "sr", "hr", "sl"]
        let complexScriptOrTonal = ["zh", "ja", "ko", "ar", "hi", "th", "vi", "fa", "he", "ur"]
        
        // Get available models for the specified device
        let currentSupport = deviceName != nil ? modelSupport(for: deviceName!) : modelSupport()
        let availableModels = currentSupport.supported
        
        // Extract primary language code if a locale-specific code is provided
        // e.g., "pt-PT" -> "pt", "zh-Hant" -> "zh"
        let inputLanguage = language.lowercased()
        let primaryLanguageCode: String
        
        if inputLanguage.contains("-") {
            primaryLanguageCode = inputLanguage.split(separator: "-").first?.lowercased() ?? inputLanguage
        } else {
            primaryLanguageCode = inputLanguage
        }
        
        // Filter models - first based on whether it's an English-only request
        var filteredModels: [String]
        
        if englishOnly.contains(primaryLanguageCode) {
            // For English, include both multilingual and English-specific models
            filteredModels = availableModels.filter { model in
                // Include all English-specific models and multilingual models
                // Exclude .{other-language} models if they exist
                !model.contains(".") || model.contains(".en")
            }
        } else {
            // For non-English, we need multilingual models only (no .en models)
            filteredModels = availableModels.filter { model in
                // Include only multilingual models (those without .en)
                !model.contains(".en")
            }
            
            // Further filter based on language complexity requirements
            if complexScriptOrTonal.contains(primaryLanguageCode) {
                // Complex scripts need at least small models, preferably large
                filteredModels = filteredModels.filter { model in
                    !model.contains("tiny") && (!model.contains("base") || model.contains("large"))
                }
            } else if mediumResourced.contains(primaryLanguageCode) {
                // Medium-resourced languages should avoid tiny models
                filteredModels = filteredModels.filter { model in
                    !model.contains("tiny")
                }
            } else if !wellResourcedEuropean.contains(primaryLanguageCode) {
                // Unknown or low-resourced languages need at least small models, similar to complex scripts
                filteredModels = filteredModels.filter { model in
                    !model.contains("tiny") && (!model.contains("base") || model.contains("large"))
                }
            }
        }
        
        // If no models passed our filters, return all multilingual models as fallback
        if filteredModels.isEmpty {
            filteredModels = availableModels.filter { !$0.contains(".en") }
        }
        
        // Only include the default model if it's in our filtered list
        let defaultModel = currentSupport.default
        let includeDefault = filteredModels.contains(defaultModel)
        
        // Order the filtered models by preference
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
    ///
    /// - Returns: An array of model names ordered by preference, with the default model first
    ///   (if available) followed by models ordered from smallest to largest.
    public func recommendedModels(forLanguages languages: [String], device deviceName: String? = nil) -> [String] {
        // Handle empty languages array
        if languages.isEmpty {
            return recommendedModels(device: deviceName)
        }
        
        // Handle single language case
        if languages.count == 1, let language = languages.first {
            return recommendedModels(forLanguage: language, device: deviceName)
        }
        
        // Get supported models for each language
        var allLanguageModels: [Set<String>] = []
        
        for language in languages {
            let supportForLanguage = recommendedModels(forLanguage: language, device: deviceName)
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
    public func downloadedRecommendedModels() throws -> [String] {
        let local = try localModels()
        let recommended = recommendedModels()
        return recommended.filter { local.contains($0) }
    }
    
    /// Returns the list of recommended models for a language that are already downloaded
    public func downloadedRecommendedModels(forLanguage language: String, device deviceName: String? = nil) throws -> [String] {
        let local = try localModels()
        let recommended = recommendedModels(forLanguage: language, device: deviceName)
        return recommended.filter { local.contains($0) }
    }
    
    /// Returns the list of recommended models that support all specified languages and are already downloaded
    ///
    /// - Parameters:
    ///   - languages: Array of language codes to find models for. Models must support ALL languages.
    ///   - deviceName: The device identifier to get model support for. Pass nil to use the current device.
    /// - Returns: Array of downloaded models that support all the specified languages
    public func downloadedRecommendedModels(forLanguages languages: [String], device deviceName: String? = nil) throws -> [String] {
        let local = try localModels()
        let recommended = recommendedModels(forLanguages: languages, device: deviceName)
        return recommended.filter { local.contains($0) }
    }
    
    // MARK: - Model Download
    
    /// Downloads a model to the local directory
    public func download(
        model: String,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> URL {
        // Download to temporary location first
        let tempModelFolder = try await huggingFaceRepo.downloadModelFiles(
            model: model,
            useBackgroundSession: useBackgroundDownloadSession,
            progressCallback: progressCallback
        )
        
        // Move to final location
        let finalModelFolder = localDirectory.appendingPathComponent(model)
        try FileManager.default.createDirectory(
            at: localDirectory,
            withIntermediateDirectories: true
        )
        
        if FileManager.default.fileExists(atPath: finalModelFolder.path) {
            try FileManager.default.removeItem(at: finalModelFolder)
        }
        
        try FileManager.default.moveItem(at: tempModelFolder, to: finalModelFolder)
        
        return finalModelFolder
    }
    
    /// Imports a model from a local file URL into the repository
    ///
    /// - Parameters:
    ///   - sourceURL: The file URL pointing to the directory containing the model files to import.
    ///                The directory itself will be copied.
    ///   - modelName: Optional name to assign to the imported model. If nil, the name of the
    ///                source directory will be used.
    ///   - overwriteExisting: If true (default), any existing model with the same name will be
    ///                        deleted before importing. If false, and a model with the same name
    ///                        already exists, the function will return the URL of the existing
    ///                        model without performing the import.
    /// - Returns: The final URL of the imported model directory within the repository.
    /// - Throws: An error if file operations (directory creation, removal, copy) fail.
    @discardableResult
    public func importModel(
        from sourceURL: URL,
        modelName name: String? = nil,
        overwriteExisting: Bool = true
    ) throws -> URL {
        // Determine the final model name
        let modelNameToUse = name ?? sourceURL.lastPathComponent

        // Create the final destination path
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

        Logging.debug("Successfully imported model \\(modelNameToUse) from \\(sourceURL.path) to \\(finalModelFolder.path)")

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
    
    /// Determines the target model name based on recommended/downloaded lists and optional size preference.
    private func determineTargetModel(recommended: [String], downloaded: [String], preferredSize: String?, defaultModel: String) -> String {
        // 1. Handle preferred size
        if let preferred = preferredSize {
            // Check if preferred size is downloaded
            if let model = downloaded.first(where: { $0.contains(preferred) }) {
                return model // Use downloaded preferred size
            }
            // Check if preferred size is recommended
            if let model = recommended.first(where: { $0.contains(preferred) }) {
                return model // Use recommended preferred size (will need download check later)
            }
            // Preferred size not found or not recommended, fall back.
            Logging.debug("Preferred size '\\(preferred)' not found in recommended models. Falling back.")
            // Fall through to logic below (use first downloaded, or first recommended)
        }

        // 2. No preferred size OR preferred size fallback - use first downloaded if available
        if let model = downloaded.first {
            return model
        }

        // 3. No preferred size/fallback, none downloaded - use first recommended or device default
        return recommended.first ?? defaultModel
    }

    /// Downloads the best model for the specified languages if needed and returns its name.
    /// - Parameters:
    ///   - languages: Array of language codes to support
    ///   - preferredSize: Optional preferred model size. If specified, will try to use a model of this size.
    ///   - progressCallback: Optional callback to track download progress
    /// - Returns: The name of the downloaded or existing model
    public func downloadedModel(
        forLanguages languages: [String],
        preferredSize: String? = nil,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> String {
        // Get recommended and downloaded models specific to the languages
        let recommended = recommendedModels(forLanguages: languages)
        let downloadedForLang = try downloadedRecommendedModels(forLanguages: languages)
        let defaultModel = modelSupport().default // Use device default as ultimate fallback

        // Determine the best model name to use
        let targetModel = determineTargetModel(
            recommended: recommended,
            downloaded: downloadedForLang,
            preferredSize: preferredSize,
            defaultModel: defaultModel
        )

        // Check if the target model is already downloaded locally (check *all* local models)
        let allDownloaded = try localModels()
        if allDownloaded.contains(targetModel) {
            return targetModel
        } else {
            // Download the target model if it's not already local
            _ = try await download(model: targetModel, progressCallback: progressCallback)
            return targetModel
        }
    }

    /// Downloads the best model for the current device if needed and returns its name.
    /// - Parameters:
    ///   - preferredSize: Optional preferred model size. If specified, will try to use a model of this size.
    ///   - progressCallback: Optional callback to track download progress
    /// - Returns: The name of the downloaded or existing model
    public func downloadedModel(
        preferredSize: String? = nil,
        progressCallback: ((Progress) -> Void)? = nil
    ) async throws -> String {
        // Get recommended and downloaded models for the current device
        let recommended = recommendedModels()
        let downloaded = try downloadedRecommendedModels()
        let defaultModel = modelSupport().default

        // Determine the best model name to use
        let targetModel = determineTargetModel(
            recommended: recommended,
            downloaded: downloaded,
            preferredSize: preferredSize,
            defaultModel: defaultModel
        )

        // Check if the target model is already downloaded locally
        if downloaded.contains(targetModel) {
            return targetModel
        } else {
            // Download the target model if it's not already local
            _ = try await download(model: targetModel, progressCallback: progressCallback)
            return targetModel
        }
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
    public static func forTesting(
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
