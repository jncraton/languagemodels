# Changelog

## 0.20

### Changed

- Add new separators to document chunking heuristic

### Fixed

- Allow missing query prefixes for embedding models

### Added

- Support GIST-small-Embedding-v0 embedding model
- Store model runtime stats to improve benchmarking and analysis

## 0.19 - 2024-04-18

### Added

- Support Meta-Llama-3-8B-Instruct
- Support gemma-2b-it
- Support h2o-danube2-1.8b-chat
- Support WizardLM-2-7B

## 0.18.0 - 2024-02-23

### Fixed

- Correct issue causing `choices` to be scored improperly

## 0.17.0 - 2024-02-15

### Added

- CUDA 12 support

## 0.16.0 - 2024-02-04

### Fixed

- Run embedding models on CPU to work around memory copy issue

## 0.15.0 - 2024-02-04

### Changed

- Improve embedding search performance

### Added

- Add openchat-3.5-0106 model
- Add h2o-danube-1.8b-chat model

## 0.14.0 - 2024-01-06

### Changed

- Simplified dialogstudio system message

### Fixed

- Correct missing instruction in openchat prompt

## 0.13.0 - 2024-01-05

### Changed

- Improved search speed when searching many documents
- Reduce memory usage for large document embeddings
- Updated to TinyLlama Chat v1.0
- Remove auto model scaling on Colab
- Correct phi-1.5 prompt format
- Correct model license metadata

### Added

- Add Mistral-7B-Instruct-v0.2 model
- Add openchat-3.5-1210 model
- Add phi-2 model
- Support static batching by passing lists to `do`
- Support choices list on `do` to restrict possible outputs

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