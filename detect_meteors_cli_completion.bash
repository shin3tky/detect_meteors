# Bash completion for detect_meteors_cli.py
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
          -t --target
          -o --output
          --debug-dir
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
          --no-parallel
          --profile
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
          --output-overwrite"

    # Provide directory completion for path arguments
    case "${prev}" in
        -t|--target|-o|--output|--debug-dir)
            # Complete with directories
            COMPREPLY=( $(compgen -d -- "${cur}") )
            return 0
            ;;
        --progress-file)
            # Complete with .json files
            COMPREPLY=( $(compgen -f -X '!*.json' -- "${cur}") )
            return 0
            ;;
        --focal-factor)
            # Suggest common sensor types
            local sensor_types="MFT APS-C APS-C_CANON APS-H FF FULLFRAME 1INCH"
            COMPREPLY=( $(compgen -W "${sensor_types}" -- "${cur}") )
            return 0
            ;;
        --sensor-type)
            # Suggest available sensor type presets
            local sensor_types="MFT APS-C APS-C_CANON APS-H FF FULLFRAME 1INCH"
            COMPREPLY=( $(compgen -W "${sensor_types}" -- "${cur}") )
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
complete -F _detect_meteors_cli_completion detect_meteors_cli.py
complete -F _detect_meteors_cli_completion ./detect_meteors_cli.py
complete -F _detect_meteors_cli_completion python detect_meteors_cli.py
complete -F _detect_meteors_cli_completion python3 detect_meteors_cli.py
