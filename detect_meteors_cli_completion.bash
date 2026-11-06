# Bash completion for detect-meteors and detect_meteors_cli.py (v1.6.11)
# 
# Installation:
#   1. Copy this file to /etc/bash_completion.d/ or ~/.bash_completion.d/
#   2. Or source it in your ~/.bashrc:
#      source /path/to/detect_meteors_cli_completion.bash
#   3. Restart your shell or run: source ~/.bashrc

_detect_meteors_cli_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # All command-line options
    opts="--version -h --help
          --locale
          --config
          --analyze-image
          --reference-image
          --analysis-sequence
          --analysis-progress
          --tune-params
          --tuning-config
          --analysis-region
          --comparison-sequence
          --known-meteor
          --analysis-output
          --analysis-proposal
          -t --target
          -o --output
          --debug-dir
          --debug-image
          --no-debug-image
          --input-loader
          --input-loader-config
          --detector
          --detector-config
          --output-handler
          --output-handler-config
          --hooks
          --hook-config
          --diff-threshold
          --min-area
          --min-aspect-ratio
          --hough-threshold
          --hough-min-line-length
          --hough-max-line-gap
          --min-line-score
          --no-roi
          --roi
          --workers
          --batch-size
          --auto-batch-size
          --no-auto-batch-size
          --parallel
          --no-parallel
          --profile
          --verbose
          --save-diagnostic
          --validate-raw
          --progress-file
          --no-resume
          --remove-progress
          --auto-params
          --sensor-type
          --focal-length
          --focal-factor
          --sensor-width
          --pixel-pitch
          --list-sensor-types
          --show-exif
          --show-npf
          --output-overwrite
          --fisheye"

    # Provide directory completion for path arguments
    case "${prev}" in
        -t|--target|-o|--output|--debug-dir|--analysis-sequence|--comparison-sequence|--analysis-output)
            # Complete with directories
            COMPREPLY=( $(compgen -d -- "${cur}") )
            return 0
            ;;
        --config|--input-loader-config|--detector-config|--output-handler-config|--hook-config|--tuning-config)
            # Complete with config files
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --progress-file|--analysis-progress)
            # Complete with .json files
            COMPREPLY=( $(compgen -f -X '!*.json' -- "${cur}") )
            return 0
            ;;
        --analyze-image|--reference-image|--known-meteor)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --save-diagnostic)
            # Complete with files for diagnostic output
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --focal-factor)
            # Suggest common sensor types (ordered by sensor size)
            local sensor_types="1INCH MFT APS-C APS-C_CANON APS-H FF FULLFRAME MF44X33 MF54X40 0.64 0.79 1.0 1.3 1.5 1.6 2.0 2.7"
            COMPREPLY=( $(compgen -W "${sensor_types}" -- "${cur}") )
            return 0
            ;;
        --sensor-type)
            # Suggest available sensor type presets (ordered by sensor size)
            local sensor_types="1INCH MFT APS-C APS-C_CANON APS-H FF FULLFRAME MF44X33 MF54X40"
            COMPREPLY=( $(compgen -W "${sensor_types}" -- "${cur}") )
            return 0
            ;;
        --locale)
            # Suggest available locales
            local locales="en ja"
            COMPREPLY=( $(compgen -W "${locales}" -- "${cur}") )
            return 0
            ;;
        --diff-threshold|--min-area|--hough-threshold|--hough-min-line-length|--hough-max-line-gap|--workers|--batch-size)
            # These expect integer values - no completion
            return 0
            ;;
        --min-aspect-ratio|--min-line-score|--focal-length|--sensor-width|--pixel-pitch)
            # These expect float values - no completion
            return 0
            ;;
        --roi)
            # ROI format hint
            COMPREPLY=( "x1,y1;x2,y2;x3,y3" )
            return 0
            ;;
        --analysis-region)
            COMPREPLY=( "x0,y0,x1,y1" )
            return 0
            ;;
        --analysis-proposal)
            COMPREPLY=( $(compgen -W "baseline 1 2 3 4 5" -- "${cur}") )
            return 0
            ;;
    esac

    # Complete with available options
    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
        return 0
    fi

    # Default to directory completion
    COMPREPLY=( $(compgen -d -- "${cur}") )
    return 0
}

# Register the completion function
complete -F _detect_meteors_cli_completion detect-meteors
complete -F _detect_meteors_cli_completion detect_meteors_cli.py
complete -F _detect_meteors_cli_completion ./detect_meteors_cli.py
