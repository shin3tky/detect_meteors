# Roadmap

## Version 1.x - Foundation and Auto-Parameter Optimization

### Core Detection System
- [x] **v1.0.1** - Initial release with manual parameter configuration
- [x] **v1.0.2** - Enhanced help descriptions for Hough transform parameters
- [x] **v1.0.3** - RAW file validation and automatic batch-size tuning

### Progress Management
- [x] **v1.1.0** - Resumable processing with JSON progress tracking
- [x] **v1.1.0** - Safe Ctrl-C interruption handling

### Auto-Parameter Estimation (Image-based)
- [x] **v1.2.0** - Initial `diff_threshold` auto-estimation (3-sigma rule)
- [x] **v1.2.1** - Improved percentile-based `diff_threshold` estimation
- [x] **v1.3.1** - Complete auto-parameter estimation for all three parameters
  - [x] Star size distribution analysis for `min_area`
  - [x] Image geometry-based `min_line_score` estimation
  - [x] Focal length support for lens-specific optimization

### Auto-Parameter Optimization (Physics-based)
- [x] **v1.4.1** - NPF Rule-based scientific optimization (MILESTONE)
  - [x] EXIF metadata extraction from RAW files
  - [x] NPF Rule integration for exposure time validation
  - [x] Star trail physics estimation (Earth's rotation)
  - [x] Meteor speed physics (3× faster than stars)
  - [x] Shooting condition quality scoring
  - [x] Sensor characterization (pixel pitch calculation)

### Safety Enhancement
- [x] **v1.4.2** - Output file protection and safety features
  - [x] Skip overwriting existing output files by default
  - [x] `--output-overwrite` flag for explicit overwrite permission
  - [x] Warning and exit when target and output directories are identical

### Sensor Type Presets
- [x] **v1.5.0** - Simplified NPF configuration with sensor presets
  - [x] Unified `SENSOR_PRESETS` configuration (MFT, APS-C, APS-C_CANON, APS-H, FF, 1INCH)
  - [x] New `--sensor-type` option for one-stop sensor configuration
  - [x] New `--list-sensor-types` option to display available presets
  - [x] Parameter override priority (individual args > preset > default)
  - [x] Updated shell completion scripts (bash/zsh)
  - [x] Backward compatibility with v1.4.x options

### Medium Format Support
- [x] **v1.5.1** - Medium format sensor support
  - [x] MF44X33 preset (Fujifilm GFX, Pentax 645Z, Hasselblad X2D)
  - [x] MF54X40 preset (Hasselblad H6D-100c)
  - [x] Sensor size ordering (smallest to largest)
  - [x] Updated shell completion scripts with medium format types

### Sensor Override Validation
- [x] **v1.5.2** - Automatic validation for sensor preset overrides
  - [x] Warning when `--sensor-width` deviates more than ±30% from preset
  - [x] Warning when `--pixel-pitch` deviates more than ±50% from preset
  - [x] Informational warnings only (processing continues normally)
  - [x] Enhanced `apply_sensor_preset()` with validation support
  - [x] Comprehensive test coverage (23 test cases)

### Fisheye Lens Correction
- [x] **v1.5.3** - Fisheye lens support with equisolid angle projection
  - [x] `--fisheye` flag for fisheye correction
  - [x] Position-dependent effective focal length calculation
  - [x] NPF calculation based on edge (worst case) focal length
  - [x] Star trail ratio adjustment for image edges
  - [x] Extensible projection model infrastructure
  - [x] Comprehensive test coverage (27 test cases)

### ROI Display and Documentation
- [x] **v1.5.4** - ROI display improvement and documentation
  - [x] Brightened ROI selection image for dark conditions
  - [x] NOTICE document for third-party attributions

### Code Architecture Refactoring
- [x] **v1.5.5** - Modular code structure for v2.x preparation
  - [x] Separated CLI interface from core logic
  - [x] Created `meteor_core/` package with modular components
  - [x] Introduced `detectors/` subpackage with base class
  - [x] Introduced `outputs/` subpackage for result handling
  - [x] Enhanced type safety with TypedDict
  - [x] Maintained full backward compatibility

### Input/Output Plugin Infrastructure
- [x] **v1.5.6** - Plugin-ready input/output foundation
  - [x] `InputLoader`/`MetadataExtractor` protocols with dataclass/Pydantic helper bases
  - [x] Built-in `RawImageLoader` plugin plus discovery via entry points and standardized plugin directories (`~/.detect_meteors/input_plugins`, `~/.detect_meteors/detector_plugins`, `~/.detect_meteors/output_plugins`; input path renamed from `~/.detect_meteors/plugins` in v1.5.11)
  - [x] `PipelineConfig` and `DetectionPipeline` protocol to centralize orchestration and loader resolution
  - [x] `OutputHandler` protocol to standardize candidate/debug persistence

### Progress Metadata & Quality Tooling
- [x] **v1.5.7** - Enhanced `progress.json` with CLI parameters, ROI, and processing metadata
- [x] **v1.5.8** - Added flake8 linting to complement Black formatting

### Packaging Modernization
- [x] **v1.5.9** - Migrated project configuration to PEP 621 `pyproject.toml`

### Plugin Architecture Hardening
- [x] **v1.5.10** - Migrated plugin interfaces from Protocol to ABC base classes
- [x] **v1.5.11** - Unified plugin registry behavior and standardized plugin directories

### Remaining Goals for v1.x (-2025 4Q)
- [ ] Camera model database for automatic sensor detection from EXIF
- [ ] Per-image adaptive parameter adjustment
- [ ] Declination support with GPS coordinate extraction
- [ ] Advanced quality metrics (focus quality, atmospheric transparency)
- [ ] Enhanced stability and error handling

## Version 2.x - Architecture and Extensibility

- 2026 1Q-
- Implementation of plugin architecture (building on v1.5.5 modular structure)
- Modular detection pipeline with swappable detectors
- Custom filter and processor support
- Third-party integration capabilities
- Configuration file support for detector plugins

## Version 3.x - Intelligence and Learning

- 2026 2Q-
- Integration of Machine Learning-based detection
- Training on labeled meteor datasets
- Advanced pattern recognition
- Adaptive learning from user feedback

---

**Current Status**: v1.5.11 (Plugin registry consistency and metadata)
**Next Focus**: Enhanced stability, plugin ecosystem hardening, and error handling
