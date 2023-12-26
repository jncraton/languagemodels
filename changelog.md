# Changelog

## 0.13.0

### Changed

- Improved search speed when searching many documents
- Reduce memory usage for large document embeddings

## 0.12.0 - 2023-12-02

### Changed

- Remove explicit setuptools dependency (see [CTranslate2#1526](https://github.com/OpenNMT/CTranslate2/pull/1526))

### Fixed

- Reduce model size when not using a CPU in Colab

## 0.11.0 - 2023-12-02

### Changed

- Default to 8GB model size on Colab
- Allow 2048 token response by default on Colab
- Use Colab GPU by default if available
- Skip returning prompt for decoder-only models
- Ensure whitespace is removed from decoder-only outputs

### Added

- Add neural-chat-7b-v3-1 as default 8GB model
- Add max_tokens config option

## 0.10.0 - 2023-10-29

### Added

- Add gte-tiny embedding model
- Properly support Python 3.12

### Fixed

- Removed extra classification prompt when performing classification with generative models
- Prevent doubling of special tokens during classification

## 0.9.0 - 2023-10-07

### Changed

- Use per-model instruction formats
- Batch chunk embeddings for faster performance embedding larger documents

### Added

- Automatically use query prefixes as needed for embeddings
- Add phi-1.5 model
- Add dialogstudio base model
- Add support for gte-small embeddings
- Add support for bge-small-en embeddings

### Fixed

- Allow token suppression on decoder-only models
- Remove HTML comments appearing in some wiki pages

## 0.8.0 - 2023-08-04

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