//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.

import Foundation

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
    public let useBackgroundDownloadSession: Bool
    
    /// Indicates whether the remote configuration has been loaded
    public private(set) var isRemoteConfigLoaded: Bool = false
    
    /// The configuration for model support - starts with fallback and updates when remote config loads
    private var _modelSupportConfig: ModelSupportConfig
    
    /// Task that handles the async loading of the remote configuration
    private var configLoadingTask: Task<Void, Never>?
    
    /// Provides access to the current model support configuration
    public var modelSupportConfig: ModelSupportConfig {
        return _modelSupportConfig
    }
    
    /// Returns the default local directory for storing models based on the repository identifier
    public static func defaultLocalDirectory(for repoIdentifier: String) -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent("huggingface/models/\(repoIdentifier)")
    }
    
    // MARK: - Initialization
    
    /// Initializes a new ModelRepo instance synchronously with fallback configuration
    /// and starts an async task to fetch the remote configuration.
    public init(
        huggingFaceRepo: HuggingFaceRepo = .init(),
        localDirectory: URL? = nil,
        useBackgroundDownloadSession: Bool = false
    ) {
        self.huggingFaceRepo = huggingFaceRepo
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
        
        // Use provided directory or create default based on the repo
        self.localDirectory = localDirectory ?? ModelRepo.defaultLocalDirectory(for: huggingFaceRepo.identifier)
        
        // Start with fallback configuration
        self._modelSupportConfig = Constants.fallbackModelSupportConfig
        
        // Start async task to fetch remote configuration
        configLoadingTask = Task {
            do {
                let remoteConfig = try await huggingFaceRepo.fetchModelSupportConfig()
                self._modelSupportConfig = remoteConfig
                self.isRemoteConfigLoaded = true
                Logging.debug("ModelRepo successfully loaded remote configuration")
            } catch {
                Logging.error("Error fetching remote config: \(error). Using fallback configuration.")
            }
        }
    }
    
    /// Wait for the remote configuration to be loaded if it's still in progress
    public func waitForRemoteConfig() async {
        if let task = configLoadingTask {
            await task.value
        }
    }
    
    /// Register a completion handler to be called when the remote configuration is loaded
    public func whenRemoteConfigLoaded(completion: @escaping (ModelSupportConfig) -> Void) {
        // If already loaded, call completion immediately
        if isRemoteConfigLoaded {
            completion(modelSupportConfig)
            return
        }
        
        // Otherwise, start a task to wait and then call completion
        Task {
            await waitForRemoteConfig()
            completion(modelSupportConfig)
        }
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
    private func modelSupport() -> ModelSupport {
        modelSupportConfig.modelSupport(for: Self.deviceName())
    }
    
    /// Returns the model support for a specific device
    private func modelSupport(for deviceName: String) -> ModelSupport {
        modelSupportConfig.modelSupport(for: deviceName)
    }
    
    /// Returns the recommended models for a device
    /// 
    /// - Parameter deviceName: The device identifier to get model support for.
    ///   Pass nil to use the current device.
    /// - Returns: A `ModelSupport` object with default and supported models for the device
    public func recommendedModels(device deviceName: String? = nil) -> ModelSupport {
        if let deviceName = deviceName {
            return modelSupport(for: deviceName)
        } else {
            return modelSupport()
        }
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
    /// - Returns: A `ModelSupport` object containing the default and supported models for the specified language
    public func recommendedModels(forLanguage language: String, device deviceName: String? = nil) -> ModelSupport {
        // Language categories by resource requirements and script complexity
        let englishOnly = ["en", "english"]
        let wellResourcedEuropean = ["es", "fr", "de", "it", "pt", "nl", "pl", "ro", "ca", "sv", "no", "da"]
        let mediumResourced = ["ru", "uk", "cs", "fi", "hu", "el", "tr", "bg", "sr", "hr", "sl"]
        let complexScriptOrTonal = ["zh", "ja", "ko", "ar", "hi", "th", "vi", "fa", "he", "ur"]
        let lowResourced = ["sw", "am", "af", "as", "bn", "bs", "cy", "eo", "et", "eu", "gl", "gn", "gu", "ha", 
                           "haw", "hy", "ig", "is", "jw", "ka", "kk", "km", "kn", "ku", "ky", "lb", "ln", 
                           "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", 
                           "oc", "pa", "ps", "sd", "si", "sk", "sl", "sn", "so", "sq", "su", "ta", "te", 
                           "tg", "tk", "tl", "tt", "uz", "yi", "yo", "zu"]
                           
        // Get available models for the specified device
        let currentSupport = recommendedModels(device: deviceName)
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
            if complexScriptOrTonal.contains(primaryLanguageCode) || lowResourced.contains(primaryLanguageCode) {
                // Complex scripts or low-resourced languages need at least small models, preferably large
                filteredModels = filteredModels.filter { model in
                    !model.contains("tiny") && (!model.contains("base") || model.contains("large"))
                }
            } else if mediumResourced.contains(primaryLanguageCode) {
                // Medium-resourced languages should avoid tiny models
                filteredModels = filteredModels.filter { model in
                    !model.contains("tiny")
                }
            } else if wellResourcedEuropean.contains(primaryLanguageCode) {
                // Well-resourced European languages can use any model size, but prefer larger models
                // No additional filtering required, but we explicitly check the category
                // to avoid the warning about unused variable
            }
        }
        
        // If no models passed our filters, return all multilingual models as fallback
        if filteredModels.isEmpty {
            filteredModels = availableModels.filter { !$0.contains(".en") }
        }
        
        // Use the default model from config if it's in our filtered list, otherwise pick the first filtered model
        let defaultModel = filteredModels.contains(currentSupport.default) 
            ? currentSupport.default 
            : filteredModels.first ?? currentSupport.default
        
        return ModelSupport(
            default: defaultModel,
            supported: filteredModels
        )
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
    /// - Returns: A `ModelSupport` object containing models that support all specified languages,
    ///   with the default being the smallest adequate model.
    public func recommendedModels(forLanguages languages: [String], device deviceName: String? = nil) -> ModelSupport {
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
            allLanguageModels.append(Set(supportForLanguage.supported))
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
        let supportedModels = currentSupport.supported.filter { intersection.contains($0) }
        
        // If no models support all languages, return the device default
        if supportedModels.isEmpty {
            return currentSupport
        }
        
        // The first model in supportedModels is the smallest one that supports all languages
        return ModelSupport(
            default: supportedModels.first ?? currentSupport.default,
            supported: supportedModels
        )
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
        let recommended = recommendedModels().supported
        return local.filter { recommended.contains($0) }
    }
    
    /// Returns the list of recommended models for a language that are already downloaded
    public func downloadedRecommendedModels(forLanguage language: String, device deviceName: String? = nil) throws -> [String] {
        let local = try localModels()
        let recommended = recommendedModels(forLanguage: language, device: deviceName).supported
        return local.filter { recommended.contains($0) }
    }
    
    /// Returns the list of recommended models that support all specified languages and are already downloaded
    ///
    /// - Parameters:
    ///   - languages: Array of language codes to find models for. Models must support ALL languages.
    ///   - deviceName: The device identifier to get model support for. Pass nil to use the current device.
    /// - Returns: Array of downloaded models that support all the specified languages
    public func downloadedRecommendedModels(forLanguages languages: [String], device deviceName: String? = nil) throws -> [String] {
        // Get locally downloaded models
        let local = try localModels()
        
        // Get models that support all specified languages
        let recommended = recommendedModels(forLanguages: languages, device: deviceName).supported
        
        // Return the intersection - models that are both downloaded and support all languages
        return local.filter { recommended.contains($0) }
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
    
    /// Deletes a downloaded model
    public func delete(model: String) throws {
        let modelFolder = localDirectory.appendingPathComponent(model)
        try FileManager.default.removeItem(at: modelFolder)
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
        let modelFilters = ModelVariant.allCases.map { "\($0.description)\($0.description.contains("large") ? "" : "/")" } // Include quantized models for large
        let modelVariants = modelFiles.map { $0.components(separatedBy: "/")[0] + "/" }
        let filteredVariants = Set(modelVariants.filter { item in
            let count = modelFilters.reduce(0) { count, filter in
                let isContained = item.contains(filter) ? 1 : 0
                return count + isContained
            }
            return count > 0
        })

        let availableModels = filteredVariants.map { variant -> String in
            variant.trimmingFromEnd(character: "/", upto: 1)
        }

        // Sorting order based on enum
        let sizeOrder = ModelVariant.allCases.map { $0.description }

        let sortedModels = availableModels.sorted { firstModel, secondModel in
            // Extract the base size without any additional qualifiers
            let firstModelBase = sizeOrder.first(where: { firstModel.contains($0) }) ?? ""
            let secondModelBase = sizeOrder.first(where: { secondModel.contains($0) }) ?? ""

            if firstModelBase == secondModelBase {
                // If base sizes are the same, sort alphabetically
                return firstModel < secondModel
            } else {
                // Sort based on the size order
                return sizeOrder.firstIndex(of: firstModelBase) ?? sizeOrder.count
                    < sizeOrder.firstIndex(of: secondModelBase) ?? sizeOrder.count
            }
        }

        return sortedModels
    }
} 
