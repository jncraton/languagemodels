# Changelog

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