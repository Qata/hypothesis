#!/usr/bin/env python3

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────
# Orchestrator Script for Module-Capability-Based Porting Workflow
# Focuses on porting complete module capabilities rather than individual functions.
# Features:
# - Architectural blueprint creation for module capability analysis
# - Module capability identification and planning (not function-by-function)
# - Capability-focused test generation and implementation
# - Idiomatic Rust pattern adoption guided by architectural blueprint
# - Integration testing between module capabilities
# - Commit Agent to finalize changes per capability (user as sole author)
# - Comprehensive logging (echoed to stdout)
# - Module-aware prompts optimized for capability-based development
# - React to planner returning 'None' for no work
# - End-of-run summarisation (summary only)
# - Executes agent calls from the output directory
# ──────────────────────────────────────────────────────────────────────────

# Configuration
from pathlib import Path
AGENT_CLI = ["claude", "--dangerously-skip-permissions"]
TIMEOUT = 12000  # seconds (reduced from 30 minutes to 2 minutes)
LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "orchestrator.log"
RUST_OUTPUT_DIR = Path(__file__).parent / "conjecture-rust"
TEST_DIR = RUST_OUTPUT_DIR / "verification-tests"

# Master prompt
MASTER_PROMPT = (
    "You are a subordinate agent spawned by the Orchestrator. Follow its instructions exactly. "
    "Always respond with the required output format."
)

# Role-specific prompts
ARCHITECT_PROMPT = (
    "Architect: Analyze the entire Python codebase structure, identify key modules, classes, and design patterns. "
    "Document the overall architecture, data flow, and interdependencies. Recommend idiomatic Rust equivalents "
    "for Python patterns (e.g., duck typing → traits, inheritance → composition, etc.). "
    "Create an architectural blueprint that guides language-appropriate porting rather than literal translation. "
    "Output a comprehensive architectural analysis and porting strategy."
)
MODULE_EXTRACTOR_PROMPT = (
    "ModuleExtractor: Based on the architectural blueprint, identify the core functional modules that should be ported. "
    "Extract module names that represent coherent capabilities (e.g., 'ChoiceSystem', 'EngineOrchestrator', 'TreeStructures'). "
    "Return ONLY a newline-separated list of module names, one per line. No descriptions, no numbers, just module names. "
    "Focus on logical modules, not necessarily Python file names."
)
PLANNER_PROMPT_INITIAL = (
    "Create a capability-focused TODO list for the specific TARGET MODULE by analyzing the provided Python codebase, "
    "scanning local Rust files for partial ports, and referencing the architectural blueprint. "
    "Break down the TARGET MODULE's work into coherent CAPABILITIES, not individual functions. "
    "Each item should be a complete capability that provides a coherent set of functionality within this module "
    "(e.g., 'Float encoding/decoding system', 'Choice constraint validation', 'Tree node management'). "
    "Focus on what the TARGET MODULE needs to DO, not how the Python code does it. "
    "Number each item clearly (1. 2. 3. etc.) with the capability name and its responsibilities. "
    "Respond only with the numbered TODO list of capabilities for this TARGET MODULE. Respond with 'None' if there's nothing left to do."
)
PLANNER_PROMPT_REVISION = (
    "Planner: Revise the existing plan and TODO list based on the original plan and the following QA feedback. "
    "Respond only with the updated plan. Respond with 'None' if there's nothing left to do."
)
TEST_PROMPT = (
    "TestGenerator: Create comprehensive tests for the specific MODULE CAPABILITY being worked on. "
    "Focus on testing the complete capability's behavior, not individual functions. "
    "Using PyO3 and FFI, write integration tests that validate the entire capability works correctly. "
    "Tests should validate the capability's core responsibilities and interface contracts. "
    "Follow the architectural blueprint for idiomatic Rust test patterns. "
    "Output only the test code for this specific module capability."
)
CODER_PROMPT = (
    "Coder: Implement the complete MODULE CAPABILITY specified in the current task. "
    "Design and implement the entire capability as a cohesive Rust module using idiomatic patterns from the architectural blueprint. "
    "Focus on the capability's core responsibilities and provide a clean, well-defined interface. "
    "Use appropriate Rust patterns (traits, enums, error handling) rather than direct Python translation. "
    "Add debug logging, use uppercase hex notation where applicable. "
    "Output the complete Rust implementation for this module capability."
)
VERIFIER_PROMPT = (
    "Verifier: Focus ONLY on the specific MODULE CAPABILITY being worked on. "
    "Run the tests for this capability and ensure the Rust implementation provides the expected behavior and interface. "
    "Verify that the capability integrates properly with other modules and follows the architectural blueprint. "
    "Apply minimal modifications to make the capability tests pass while maintaining idiomatic Rust patterns. "
    "Output only the verification results and any code adjustments needed for this specific capability."
)
QA_PROMPT = (
    "QA: Review the code changes for the current MODULE CAPABILITY against the architectural blueprint and tests. "
    "If the capability implementation has issues, find exactly ONE specific fault (correctness, design, or architectural compliance). "
    "If the current capability is correctly implemented, check what's missing from the overall porting plan. "
    "Reply with one of: "
    "'OK' if the entire module porting is complete and all capabilities are implemented, "
    "'NEXT' if the current capability is done and you want to move to the next capability, "
    "or 'FAULT: [description]' with exactly ONE specific fault to fix in the current capability."
)
COMMIT_PROMPT = (
    "CommitAgent: Stage all changes in the Rust output directory and create a git commit with a descriptive message. "
    "Treat the user as the sole author—do not mention co-authors. Output only the commit message."
)
SUMMARISER_PROMPT = (
    "Summariser: Provide a concise summary of the orchestrator log, highlighting key decisions and failures."
)

# Helpers

def ensure_directories():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str):
    timestamp = datetime.utcnow().isoformat() + "Z"
    line = f"[{timestamp}] {message}"
    
    try:
        with LOG_FILE.open("a", encoding="utf-8", buffering=1) as f:  # Line buffering
            f.write(line + "\n")
            f.flush()  # Force write to disk
    except UnicodeEncodeError as e:
        # Handle encoding issues by writing the error and repr of the message
        try:
            safe_line = f"[{timestamp}] [ENCODING ERROR] {repr(message)}"
            with LOG_FILE.open("a", encoding="utf-8", buffering=1) as f:
                f.write(safe_line + "\n")
                f.flush()
            print(f"Warning: Encoding error in log message: {e}")
        except Exception:
            print(f"Critical: Failed to log message due to encoding: {e}")
    except Exception as e:
        print(f"Warning: Failed to write to log file: {e}")
    
    print(line)


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def prompt(child_snippet: str) -> str:
    return f"{MASTER_PROMPT}\n{child_snippet}"


def wait_until_next_hour():
    """Wait until 1 minute past the next hour"""
    now = datetime.now()
    next_hour = now.replace(minute=1, second=0, microsecond=0) + timedelta(hours=1)
    wait_seconds = (next_hour - now).total_seconds()
    log(f"Waiting {wait_seconds/60:.1f} minutes until {next_hour.strftime('%H:%M')}")
    time.sleep(wait_seconds)


def run_agent(child_snippet: str, stdin_input: str) -> str:
    full_prompt = prompt(child_snippet)
    agent_name = child_snippet.split(':')[0] if ':' in child_snippet else child_snippet[:20]
    log(f"RUN AGENT: {agent_name}")
    log(f"Input size: {len(stdin_input)} characters")
    
    try:
        proc = subprocess.run(
            AGENT_CLI + ["-p", full_prompt],
            input=stdin_input,
            text=True,
            capture_output=True,
            timeout=TIMEOUT,
            check=True,
            cwd=RUST_OUTPUT_DIR
        )
        output = proc.stdout.strip()
        
        # Check for Claude API usage limit message
        if output.startswith("Claude AI usage limit reached"):
            log("Ran out of usage, will try again at the hour")
            wait_until_next_hour()
            # Recursive call with same arguments
            return run_agent(child_snippet, stdin_input)
        
        log(f"OUTPUT: {output}\n--- End of output ---")
        return output
        
    except subprocess.TimeoutExpired:
        log(f"Agent {agent_name} timed out after {TIMEOUT} seconds")
        raise
    except subprocess.CalledProcessError as e:
        # Check stderr for usage limit message too
        error_output = e.stderr or ""
        if error_output.startswith("Claude AI usage limit reached"):
            log("Ran out of usage, will try again at the hour")
            wait_until_next_hour()
            # Recursive call with same arguments
            return run_agent(child_snippet, stdin_input)
        
        # If not a usage limit error, re-raise
        raise


def create_architectural_blueprint(folder: Path) -> str:
    log(f"\n=== Creating Architectural Blueprint ===")
    
    # Gather all Python files and their contents
    python_files = [f for f in sorted(folder.glob('**/*.py')) if f.name != '__init__.py']
    codebase_content = []
    
    for py_file in python_files:
        rel_path = py_file.relative_to(folder)
        content = read_file(py_file)
        codebase_content.append(f"=== {rel_path} ===\n{content}\n")
    
    full_codebase = "\n".join(codebase_content)
    log(f"Analyzing {len(python_files)} Python files for architectural blueprint")
    
    blueprint = run_agent(ARCHITECT_PROMPT, full_codebase)
    log("Architectural blueprint created")
    return blueprint


def extract_modules_from_blueprint(blueprint: str) -> list:
    log(f"\n=== Extracting Modules from Blueprint ===")
    
    module_list_text = run_agent(MODULE_EXTRACTOR_PROMPT, blueprint)
    modules = [line.strip() for line in module_list_text.split('\n') if line.strip()]
    
    log(f"Extracted {len(modules)} modules:")
    for i, module in enumerate(modules, 1):
        log(f"  Module {i}: {module}")
    
    return modules


def parse_todo_capabilities(plan_text: str) -> list:
    """Parse numbered TODO capabilities from planner output"""
    capabilities = []
    lines = plan_text.strip().split('\n')
    current_capability = ""
    
    for line in lines:
        # Check if line starts with a number followed by period or parenthesis
        if line.strip() and (line.strip()[0].isdigit() and ('.' in line[:5] or ')' in line[:5])):
            if current_capability:
                capabilities.append(current_capability.strip())
            current_capability = line
        elif current_capability:
            current_capability += "\n" + line
    
    if current_capability:
        capabilities.append(current_capability.strip())
    
    return capabilities


def process_module(module_name: str, architectural_blueprint: str, full_python_codebase: str):
    log(f"\n=== Processing Module: {module_name} ===")
    log(f"Working on module capability: {module_name}")

    # Include architectural blueprint and full codebase in planner input
    planner_input = f"ARCHITECTURAL BLUEPRINT:\n{architectural_blueprint}\n\nTARGET MODULE:\n{module_name}\n\nFULL PYTHON CODEBASE:\n{full_python_codebase}"
    plan = run_agent(PLANNER_PROMPT_INITIAL, planner_input)
    if plan.strip() == 'None':
        log(f"Planner returned 'None'. No work needed for module {module_name}.")
        return

    # Parse individual TODO capabilities
    todo_capabilities = parse_todo_capabilities(plan)
    log(f"Found {len(todo_capabilities)} capabilities to implement:")
    for i, capability in enumerate(todo_capabilities, 1):
        log(f"  Capability {i}: {capability.split('.', 1)[1].strip() if '.' in capability else capability}")

    # Process each capability individually
    for capability_num, current_capability in enumerate(todo_capabilities, 1):
        log(f"\n--- Processing Capability {capability_num}/{len(todo_capabilities)} ---")
        log(f"Current capability: {current_capability}")

        # Create context for this specific capability
        capability_context = f"ARCHITECTURAL BLUEPRINT:\n{architectural_blueprint}\n\nTARGET MODULE:\n{module_name}\n\nCURRENT CAPABILITY TO IMPLEMENT:\n{current_capability}\n\nFULL PYTHON CODEBASE:\n{full_python_codebase}"

        capability_iteration = 0
        while True:
            capability_iteration += 1
            log(f"Capability {capability_num} - Attempt {capability_iteration}")

            try:
                test_out = run_agent(TEST_PROMPT, capability_context)
                coder_out = run_agent(CODER_PROMPT, capability_context)
            except Exception as e:
                log(f"Error during agent execution: {e}")
                capability_context += f"\n\nERROR FEEDBACK:\nAgent execution failed: {e}"
                continue

            try:
                verifier_input = f"CAPABILITY CONTEXT:\n{capability_context}\n\nTESTS:\n{test_out}\n\nCODE:\n{coder_out}"
                verifier_out = run_agent(VERIFIER_PROMPT, verifier_input)

                qa_input = f"CAPABILITY:\n{current_capability}\n\nCODE CHANGES:\n{coder_out}\n\nVERIFICATION:\n{verifier_out}\n\nOVERALL PLAN:\n{plan}"
                qa_out = run_agent(QA_PROMPT, qa_input)
            except Exception as e:
                log(f"Error during verification/QA: {e}")
                capability_context += f"\n\nERROR FEEDBACK:\nVerification/QA failed: {e}"
                continue

            if qa_out.strip() == "OK":
                log(f"✓ QA declares entire module complete. Finishing module processing.")
                try:
                    commit_msg = run_agent(COMMIT_PROMPT, f"Completed entire module: {module_name}")
                    log(f"COMMIT MESSAGE: {commit_msg}")
                except Exception as e:
                    log(f"Warning: Commit failed: {e}")
                return  # Exit the entire module processing
            elif qa_out.strip() == "NEXT":
                log(f"✓ Capability {capability_num} completed. QA requests moving to next capability.")
                try:
                    commit_msg = run_agent(COMMIT_PROMPT, f"Completed capability: {current_capability}")
                    log(f"COMMIT MESSAGE: {commit_msg}")
                except Exception as e:
                    log(f"Warning: Commit failed: {e}")
                break  # Exit current capability loop, continue to next capability
            elif qa_out.strip().startswith("FAULT:"):
                fault_description = qa_out.strip()[6:].strip()  # Remove "FAULT:" prefix
                log(f"✗ Capability {capability_num} needs revision: {fault_description}")
                # Update capability context with QA feedback for next iteration
                capability_context += f"\n\nQA FEEDBACK:\n{fault_description}"
            else:
                log(f"✗ Capability {capability_num} unexpected QA response: {qa_out}")
                # Treat unexpected responses as feedback
                capability_context += f"\n\nQA FEEDBACK:\n{qa_out}"

        log(f"✓ Completed capability {capability_num}: {current_capability.split('.', 1)[1].strip() if '.' in current_capability else current_capability}")


def summarise_log():
    log("Starting summarisation step.")
    try:
        full_log = read_file(LOG_FILE)
        summary = run_agent(SUMMARISER_PROMPT, full_log)
        print("=== ORCHESTRATOR SUMMARY ===")
        print(summary)
    except Exception as e:
        log(f"Warning: Summary generation failed: {e}")
        print("=== ORCHESTRATOR SUMMARY ===")
        print("Summary generation failed, check logs for details.")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-python-folder>")
        sys.exit(1)

    ensure_directories()
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    log("Orchestrator started.")

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    # Create architectural blueprint first
    try:
        architectural_blueprint = create_architectural_blueprint(folder)
    except Exception as e:
        log(f"Fatal error creating architectural blueprint: {e}")
        sys.exit(1)
    
    # Extract modules dynamically from the blueprint
    try:
        modules = extract_modules_from_blueprint(architectural_blueprint)
        if not modules:
            log("No modules extracted from blueprint. Nothing to do.")
            return
    except Exception as e:
        log(f"Fatal error extracting modules: {e}")
        sys.exit(1)
    
    # Gather full Python codebase once for all modules to reference
    # We already have this from the blueprint creation, let's reuse it
    python_files = [f for f in sorted(folder.glob('**/*.py')) if f.name != '__init__.py']
    codebase_content = []
    for py_file in python_files:
        rel_path = py_file.relative_to(folder)
        content = read_file(py_file)
        codebase_content.append(f"=== {rel_path} ===\n{content}\n")
    full_python_codebase = "\n".join(codebase_content)
    
    # Process each derived module
    for module_name in modules:
        try:
            process_module(module_name, architectural_blueprint, full_python_codebase)
        except Exception as e:
            log(f"Error processing module {module_name}: {e}")
            log("Continuing with next module...")

    log("All modules processed.\n")
    summarise_log()

if __name__ == "__main__":
    main()
