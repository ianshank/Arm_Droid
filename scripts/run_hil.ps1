#Requires -Version 5.1
<#
.SYNOPSIS
    Convenience wrapper for hardware-in-the-loop (HIL) tests on Windows.

.DESCRIPTION
    Sets the required environment variables and delegates to pytest.

    Environment variables (all optional):

    ARMDROID_HIL_RUN (default: 1)
        Set to 0 to dry-run — tests are collected but the hardware
        fixture will skip them.

    ARMDROID_HIL_PORT
        Serial port to pass to the driver.  When set, maps to
        ARMDROID_ARM__TRANSPORT__SERIAL_PORT so the port override is
        picked up by Pydantic Settings without touching any config file.
        Example: $env:ARMDROID_HIL_PORT = "COM3"

    ARMDROID_HIL_SWEEP_AMPLITUDE_RAD (default: 0.05)
        Per-joint sweep amplitude in radians for test_per_joint_sweep.

    ARMDROID_HIL_FAULT_INJECT (default: unset)
        Set to 1 to include live fault-injection tests.

.EXAMPLE
    .\scripts\run_hil.ps1

.EXAMPLE
    $env:ARMDROID_HIL_PORT = "COM3"; .\scripts\run_hil.ps1 -k test_estop

.EXAMPLE
    $env:ARMDROID_HIL_FAULT_INJECT = "1"; .\scripts\run_hil.ps1
#>

param(
    # Any extra arguments are forwarded verbatim to pytest.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$env:ARMDROID_HIL_RUN = if ($env:ARMDROID_HIL_RUN) { $env:ARMDROID_HIL_RUN } else { "1" }

if ($env:ARMDROID_HIL_PORT) {
    $env:ARMDROID_ARM__TRANSPORT__SERIAL_PORT = $env:ARMDROID_HIL_PORT
}

& python -m pytest tests/hardware -m hardware -v @PytestArgs
exit $LASTEXITCODE
