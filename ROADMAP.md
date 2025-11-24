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

### Remaining Goals for v1.x
- [ ] Camera model database for automatic sensor detection
- [ ] Per-image adaptive parameter adjustment
- [ ] Declination support with GPS coordinate extraction
- [ ] Advanced quality metrics (focus quality, atmospheric transparency)
- [ ] Enhanced stability and error handling

## Version 2.x - Architecture and Extensibility

- 2026 1Q
- Implementation of plugin architecture
- Modular detection pipeline
- Custom filter and processor support
- Third-party integration capabilities

## Version 3.x - Intelligence and Learning

- 2026 2Q
- Integration of Machine Learning-based detection
- Training on labeled meteor datasets
- Advanced pattern recognition
- Adaptive learning from user feedback

---

**Current Status**: v1.4.1 (NPF Rule-based Scientific Optimization - Milestone Release)  
**Next Focus**: Camera database and adaptive parameter adjustment
