# Changelog

## 0.9.0

### Changed

- Use per-model instruction formats

### Added

- Automatically use query prefixes as needed for embeddings
- Add phi-1.5 model
- Add dialogstudio base model
- Add support for gte-small embeddings
- Add support for bge-small-en embeddings

### Fixed

- Allow token suppression on decoder-only models

## 0.8.0

### Changed

- Model names no longer include backend and quantization info
- Default to CPU inference unless GPU enabled using `lm.config["device"]="auto"`

### Added

- Add quantization info to config and use it for memory usage calculation

### Fixed

- Increase repetition penalty to 1.3 from 1.2 to help avoid repetition in smaller models

## 0.7.0 - 2023-07-27

### Changed

- Improve semantic meaning of chunk heading
- Remove sentencepiece dependency

### Added

- Support GPT-based models
- Add `code` generation function
- Create new configuration system
- Use CUDA if available

### Fixed

- Use non-greedy sampling on `complete` function
- Decrease chance of splitting chunks on decimal points
- Correct assistant example

## 0.6.0

### Changed

- Attempt to chunk context on semantic boundaries

### Added

- Allow filtering by model license

### Fixed

- Update classification to only allow valid classes to be returned

## 0.5.0 

### Changed

- Disable beam search for faster inference

## 0.4.0

### Changed

- Normalize output
- Rename some functions

### Added

- Support xl models

## 0.2.0 

### Changed

- Less verbose chat syntax

### 0.1.0 

### Changed

- Use ctranslate2 for greater efficiency

### 0.0.0 

- Original version using HuggingFace Transformers