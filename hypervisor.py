#!/usr/bin/env python3

import subprocess
import sys
import os
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    "Always respond with the required output format. "
    "WARNING: Previous sessions suffered from severe overengineering - agents created 1300+ line files "
    "with complex architectures instead of simple Python porting. You MUST avoid this mistake. "
    "Port ONLY exact Python functionality with minimal implementation."
)

# Role-specific prompts
PYTHON_ANALYZER_PROMPT = (
    "PythonAnalyzer: Identify core Python modules and basic functionality to port. "
    "Find key modules, classes, and simple patterns in the provided path. "
    "Focus on understanding basic functionality only. "
    "Output simple analysis of Python modules and their core functions."
)
RUST_ANALYZER_PROMPT = (
    "RustAnalyzer: Check what Rust code exists and identify overengineering problems. "
    "WARNING: Previous sessions created overengineered code that needs fixing. Look for: "
    "- Files over 200 lines that should be simplified "
    "- Complex architectures not found in Python "
    "- Excessive documentation beyond Python's level "
    "- Features/patterns not in the Python original "
    "Mark overengineered code as PROBLEMS TO FIX, not achievements. "
    "Identify what's been implemented correctly vs what needs simplification or porting. "
    "Output analysis identifying both missing functionality AND overengineering to remove."
)
VERIFICATION_ANALYZER_PROMPT = (
    "VerificationAnalyzer: Analyze PyO3 behavioral parity verification coverage between Python Hypothesis and Rust implementations. "
    "IMPORTANT: PyO3 verification tests are ONLY for direct behavioral comparison - e.g., verifying that the lexicographic encoding of 2.5 is identical in both Rust and Python implementations. "
    "These are NOT general interoperability tests, but specific parity checks to ensure the Rust port behaves identically to the original Python code. "
    "Examine the Python codebase at the provided path to identify all core capabilities and functionality. "
    "Then examine the Rust codebase and specifically check the verification-tests/ directory for existing PyO3 behavioral parity tests. "
    "For each Python capability, determine: 1) Is it implemented in Rust? 2) Does it have PyO3 behavioral parity verification tests? "
    "Focus on identifying capabilities that exist in BOTH Python and Rust but are NOT currently verified for behavioral parity via PyO3 tests. "
    "Also note any gaps where Python functionality is missing behavioral parity verification entirely. "
    "Output a report of: 1) Capabilities with existing PyO3 behavioral parity verification, "
    "2) Capabilities implemented in both Python and Rust but MISSING PyO3 behavioral parity verification, "
    "3) Recommendations for priority PyO3 behavioral parity verification test creation."
)
CODE_ANALYZER_PROMPT = (
    "CodeAnalyzer: Analyze the codebase to identify all mentions of TODO, FIXME, and 'simplified' across all source files. "
    "Search through all Python and Rust source files in the provided paths to find these markers. "
    "For each mention, analyze the context to understand what functionality is missing, incomplete, or deliberately simplified. "
    "Categorize findings into: 1) Critical missing functionality that blocks core features, "
    "2) Important improvements that affect correctness or performance, "
    "3) Minor optimizations or enhancements. "
    "Provide specific file locations and line numbers for each finding. "
    "Create a prioritized list of issues that should be addressed first, focusing on items that: "
    "- Block core functionality or cause incorrect behavior "
    "- Are referenced by multiple components "
    "- Affect critical algorithms or data structures "
    "Output a succinct but complete report of missing functionality with priority recommendations."
)
ARCHITECT_PROMPT = (
    "Architect: Create simple porting plan based on analyses. "
    "SCOPE VALIDATION: Before creating plan, verify: "
    "1. Are we only porting exact Python functionality? "
    "2. Are we using simple patterns Python developers would recognize? "
    "3. Are we avoiding 'enterprise' complexity? "
    "Identify missing Python functionality in Rust and create basic porting strategy. "
    "Focus on direct translation, not architectural improvements. "
    "Output simple blueprint for porting Python to Rust."
)
MODULE_EXTRACTOR_PROMPT = (
    "ModuleExtractor: Based on the architectural blueprint, identify the core functional modules that should be ported. "
    "PRIORITY: Modules containing critical issues identified in the code analysis (TODO, FIXME, simplified) should be listed FIRST. "
    "Extract module names that represent coherent capabilities (e.g., 'ChoiceSystem', 'EngineOrchestrator', 'TreeStructures'). "
    "Order the modules by priority: 1) Modules with critical code issues first, 2) Other important modules second. "
    "Return ONLY a newline-separated list of module names, one per line. No descriptions, no numbers, just module names. "
    "Focus on logical modules, not necessarily Python file names."
)
TEST_RUNNER_PROMPT = (
    "TestRunner: Execute all available verification tests to identify current failing tests and their causes. "
    "Run the PyO3 behavioral parity verification tests in the verification-tests/ directory and any other relevant test suites. "
    "IMPORTANT: PyO3 verification tests are ONLY for direct behavioral comparison - they verify that Rust implementations produce identical outputs to Python implementations (e.g., same lexicographic encoding for number 2.5). "
    "For each failing test, analyze the failure reason and identify which specific capabilities need fixing. "
    "Focus on tests that validate exact behavioral parity between Python and Rust implementations. "
    "Output a report of: 1) All tests that are currently passing, "
    "2) All tests that are currently failing with specific error details, "
    "3) Which capabilities/modules each failing test relates to, "
    "4) Priority recommendations for which failing tests should be fixed first."
)
PLANNER_PROMPT_INITIAL = (
    "Create a simple TODO list for TARGET MODULE by finding Python functionality to port. "
    "SCOPE VALIDATION: Before planning, answer: "
    "1. Does this exact functionality exist in Python Hypothesis? "
    "2. Can this be implemented in under 200 lines? "
    "3. Would Python developers recognize this pattern? "
    "If any answer is 'no', STOP and simplify. "
    "STRICT CONSTRAINT: Plan ONLY exact Python porting. Do not add: "
    "- Additional features not in Python "
    "- Complex architectures beyond Python's approach "
    "- More than 5 capabilities per module "
    "Break work into simple capabilities matching Python structure. "
    "Number each item (1. 2. 3.) with capability name only. "
    "Respond with numbered list or 'None' if complete."
)
PLANNER_PROMPT_REVISION = (
    "Planner: Revise the existing plan and TODO list based on the original plan and the following QA feedback. "
    "Respond only with the updated plan. Respond with 'None' if there's nothing left to do."
)
TEST_PROMPT = (
    "TestGenerator: PORT EXISTING PYTHON TESTS to Rust. "
    "STRICT CONSTRAINT: Port ONLY the exact test functionality. Do not add: "
    "- Additional test cases not in Python "
    "- Test infrastructure beyond basic Rust tests "
    "- More than 100 lines per test file "
    "SUCCESS CRITERIA: "
    "- Rust tests behave identically to Python tests "
    "- No new features or 'improvements' "
    "- Minimal viable implementation "
    "Find Python test files and port directly using #[test] and assert_eq!. "
    "Output only ported test code."
)
CODER_PROMPT = (
    "Coder: PORT EXISTING PYTHON CODE to Rust. "
    "STRICT CONSTRAINT: Port ONLY the exact Python functionality. Do not add: "
    "- Additional features not in Python "
    "- Complex architectures beyond Python's simple approach "
    "- More than 200 lines per capability "
    "SUCCESS CRITERIA: "
    "- Rust code behaves identically to Python code "
    "- No new features or 'improvements' "
    "- Minimal viable implementation "
    "Before implementing, answer: Does this exact functionality exist in Python? "
    "Find Python source files and port algorithms directly using basic Rust patterns. "
    "Output only ported Rust code."
)
VERIFIER_PROMPT = (
    "Verifier: DIRECTLY COMPARE Rust outputs to Python outputs. "
    "STRICT CONSTRAINT: Create ONLY direct comparison tests. Do not add: "
    "- Wrapper infrastructure or bindings "
    "- Verification frameworks beyond simple comparison "
    "- More than 50 lines per verification "
    "SUCCESS CRITERIA: "
    "- Outputs are byte-for-byte identical "
    "- Simple direct comparison tests "
    "- Minimal code to achieve parity "
    "Use PyO3 to call Python functions and compare with Rust. "
    "Output verification results and minimal fixes only."
)
QA_PROMPT = (
    "QA: Review for DIRECT PYTHON PORTING only. "
    "SCOPE VALIDATION: Before approving, verify: "
    "1. Does this exact functionality exist in Python Hypothesis? "
    "2. Is this the simplest possible implementation? "
    "3. Would Python developers recognize this pattern? "
    "If any answer is 'no', STOP and simplify. "
    "REJECT: "
    "- More than 200 lines per capability "
    "- Features not in Python "
    "- Complex architectures "
    "- PyO3 wrapper classes "
    "CRITICAL ERROR POLICY: ALL errors are critical blocking failures: "
    "- ANY compilation errors "
    "- ANY test failures "
    "- ANY PyO3 verification failures "
    "- ANY scope violations "
    "DO NOT dismiss errors or rationalize failures. "
    "Only consider complete if: 1) Code compiles cleanly, 2) All tests pass, 3) PyO3 verification passes. "
    "Reply: 'OK', 'NEXT', or 'FAULT: [description]'."
)
DECISION_EXTRACTOR_PROMPT = (
    "DecisionExtractor: Extract the decision from the QA agent's response. "
    "The QA agent should have indicated one of: 'OK', 'NEXT', or 'FAULT: [description]'. "
    "Find this decision in the provided QA response and return ONLY the decision keyword: "
    "Return exactly 'OK', 'NEXT', or 'FAULT: [description]' - nothing else."
)
DOCUMENTATION_PROMPT = (
    "DocumentationAgent: Add basic documentation matching Python's style. "
    "STRICT CONSTRAINT: Match Python's documentation level. Do not add: "
    "- More documentation than Python equivalent "
    "- Complex architectural explanations "
    "SUCCESS CRITERIA: "
    "- Documentation ratio matches Python (10-15%) "
    "- Basic /// comments for public functions "
    "- Simple inline comments for complex logic "
    "Add minimal documentation to match Python's approach. "
    "Output summary of basic documentation added."
)
COMMIT_PROMPT = (
    "CommitAgent: Stage all changes in the Rust output directory and create a git commit with a descriptive message. "
    "Treat the user as the sole author—do not mention co-authors. Output only the commit message."
)
LOG_ANALYZER_PROMPT = (
    "LogAnalyzer: Analyze the orchestrator log to understand what work has been completed. "
    "WARNING: Previous sessions suffered from severe overengineering that wasted time and tokens. "
    "When analyzing logs, identify overengineering patterns as PROBLEMS to avoid, not successes to replicate. "
    "The log file is located at /home/ch/Develop/hypothesis/logs/orchestrator.log - read this file yourself using the Read tool. "
    "Identify which modules have been processed, which capabilities have been implemented, and what the current state is. "
    "Create a summary that captures important information from the log in organized form. "
    "Include: 1) Completed modules and their capabilities, 2) Current work in progress, "
    "3) Any failures or overengineering issues that need attention, 4) Basic metrics, "
    "5) Recommended next steps focusing on simple Python porting, 6) Timeline of major events. "
    "CRITICAL: Mark any overengineering as problems to avoid, not quality achievements."
)
SUMMARISER_PROMPT = (
    "Summariser: Provide a concise summary of the orchestrator log, highlighting key decisions and failures."
)

# Removed graceful shutdown handling - use standard signal handling

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


def parse_api_limit_timestamp(stdout: str) -> int:
    """Parse Unix timestamp from API limit message"""
    # Expected format: 'Claude AI usage limit reached|1749718800'
    if '|' in stdout:
        try:
            timestamp_str = stdout.split('|')[-1].strip()
            return int(timestamp_str)
        except (ValueError, IndexError):
            pass
    return None


def wait_for_api_limit(timestamp: int):
    """Wait until the API limit expires"""
    current_time = int(time.time())
    if timestamp <= current_time:
        log("API limit timestamp is in the past, retrying immediately")
        return
    
    wait_seconds = max(0, timestamp - current_time)  # Clamp to zero
    wait_minutes = wait_seconds / 60
    log(f"API limit until {datetime.fromtimestamp(timestamp).isoformat()}")
    log(f"Waiting {wait_minutes:.1f} minutes ({wait_seconds} seconds) for API limit to expire")
    time.sleep(wait_seconds)


def cleanup_target_directories():
    """Clean up Rust target directories to free up disk space"""
    log("Cleaning up Rust target directories...")
    
    import shutil
    
    # Find all target directories
    target_dirs = []
    for root, dirs, files in os.walk(RUST_OUTPUT_DIR):
        if 'target' in dirs:
            target_path = Path(root) / 'target'
            target_dirs.append(target_path)
    
    # Also check verification-tests target directory
    verification_target = TEST_DIR / "target"
    if verification_target.exists():
        target_dirs.append(verification_target)
    
    total_freed = 0
    for target_dir in target_dirs:
        try:
            if target_dir.exists():
                # Calculate size before deletion
                size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
                total_freed += size
                
                shutil.rmtree(target_dir)
                log(f"Deleted {target_dir} ({size // (1024*1024)} MB)")
        except Exception as e:
            log(f"Warning: Failed to delete {target_dir}: {e}")
    
    if total_freed > 0:
        log(f"Total disk space freed: {total_freed // (1024*1024)} MB")
    else:
        log("No target directories found to clean up")


def run_pyo3_verification() -> str:
    """Run PyO3 behavioral parity verification tests to validate Rust implementation produces identical outputs to Python"""
    log("Running PyO3 behavioral parity verification tests...")
    
    verification_dir = RUST_OUTPUT_DIR / "verification-tests"
    if not verification_dir.exists():
        return "SKIP: PyO3 behavioral parity verification tests not found (verification-tests directory missing)"
    
    try:
        # Build the verification tests
        log("Building PyO3 behavioral parity verification tests...")
        build_proc = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=verification_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for build
        )
        
        if build_proc.returncode != 0:
            log(f"PyO3 behavioral parity verification build failed:\n{build_proc.stderr}")
            return f"BUILD_FAILED: {build_proc.stderr}"
        
        # Run the verification tests
        log("Running PyO3 behavioral parity verification tests...")
        test_proc = subprocess.run(
            ["cargo", "run", "--release"],
            cwd=verification_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for tests
        )
        
        if test_proc.returncode == 0:
            log("✓ PyO3 behavioral parity verification tests PASSED")
            return f"PASSED: All PyO3 behavioral parity verification tests passed\n{test_proc.stdout}"
        else:
            log(f"✗ PyO3 behavioral parity verification tests FAILED:\n{test_proc.stderr}")
            return f"FAILED: PyO3 behavioral parity verification failed\n{test_proc.stderr}\n{test_proc.stdout}"
            
    except subprocess.TimeoutExpired:
        log("PyO3 behavioral parity verification tests timed out")
        return "TIMEOUT: PyO3 behavioral parity verification tests timed out after 5 minutes"
    except Exception as e:
        log(f"Error running PyO3 behavioral parity verification: {e}")
        return f"ERROR: {e}"


def run_agent(child_snippet: str, stdin_input: str, use_file_prompt: bool = False) -> str:
    full_prompt = prompt(child_snippet)
    agent_name = child_snippet.split(':')[0] if ':' in child_snippet else child_snippet[:20]
    log(f"RUN AGENT: {agent_name}")
    log(f"Input size: {len(stdin_input)} characters")
    
    # Use file-based prompt if requested or if input is very large
    if use_file_prompt or len(stdin_input) > 100000:  # 100KB threshold
        import tempfile
        import hashlib
        
        # Create a unique filename based on content hash
        content_hash = hashlib.md5((full_prompt + stdin_input).encode()).hexdigest()[:8]
        prompt_file = f"/tmp/hypervisor_prompt_{agent_name}_{content_hash}.txt"
        
        log(f"Using file-based prompt due to large input size: {prompt_file}")
        
        # Write the full context to the file
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"AGENT_PROMPT:\n{full_prompt}\n\nINPUT_CONTEXT:\n{stdin_input}")
        
        # Use a minimal prompt that just references the file
        file_prompt = f"{MASTER_PROMPT}\n{child_snippet}\n\nLARGE_CONTEXT_FILE: {prompt_file}\nPlease read the context from the file above using the Read tool."
        actual_stdin = ""
        
        try:
            proc = subprocess.run(
                AGENT_CLI + ["-p", file_prompt],
                input=actual_stdin,
                text=True,
                capture_output=True,
                timeout=TIMEOUT,
                check=True,
                cwd=RUST_OUTPUT_DIR
            )
            output = proc.stdout.strip()
            log(f"OUTPUT: {output}\n--- End of output ---")
            
            # Clean up the temporary file
            try:
                os.unlink(prompt_file)
            except:
                pass
                
            return output
        except Exception as e:
            # Clean up the temporary file on error
            try:
                os.unlink(prompt_file)
            except:
                pass
            raise e
    else:
        # Use normal prompt method
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
            log(f"OUTPUT: {output}\n--- End of output ---")
            return output
            
        except subprocess.TimeoutExpired:
            log(f"Agent {agent_name} timed out after {TIMEOUT} seconds")
            raise
        except subprocess.CalledProcessError as e:
            # Log stderr for debugging
            if e.stderr:
                log(f"Agent stderr: {e.stderr.strip()}")
            if e.stdout:
                log(f"Agent stdout: {e.stdout.strip()}")
            log(f"Agent returned exit code: {e.returncode}")
            
            # Handle different exit codes appropriately
            if e.returncode == 1:
                # Exit code 1 indicates API usage limit
                log("API usage limit reached (exit code 1)")
                if e.stdout:
                    timestamp = parse_api_limit_timestamp(e.stdout)
                    if timestamp:
                        wait_for_api_limit(timestamp)
                    else:
                        log("Could not parse API limit timestamp, waiting 5 minutes")
                        time.sleep(5 * 60)
                else:
                    log("No stdout available for timestamp parsing, waiting 5 minutes")
                    time.sleep(5 * 60)
            elif e.returncode == -9:
                # Exit code -9 indicates SIGKILL - assume prompt was too long
                if use_file_prompt:
                    log("Process killed by SIGKILL (-9) even with file-based prompt - retrying with same file approach")
                    return run_agent(child_snippet, stdin_input, use_file_prompt=True)
                else:
                    log("Process killed by SIGKILL (-9), likely due to prompt being too long")
                    log("Retrying with file-based prompt...")
                    return run_agent(child_snippet, stdin_input, use_file_prompt=True)
            else:
                # Other non-zero exit codes should result in immediate retry
                log(f"Non-API error (exit code {e.returncode}), retrying immediately")
            
            # Recursive call with same arguments
            return run_agent(child_snippet, stdin_input, use_file_prompt)


def create_architectural_blueprint(python_folder: Path, log_analysis: str = None) -> str:
    log(f"\n=== Creating Architectural Blueprint ===")
    
    # Prepare inputs for all four analyzers
    python_input = f"PYTHON_CODEBASE_PATH: {python_folder.absolute()}"
    if log_analysis:
        python_input = f"PREVIOUS_WORK_ANALYSIS:\n{log_analysis}\n\n{python_input}"
        log("Including previous work analysis in Python analysis")
    
    rust_input = f"RUST_CODEBASE_PATH: {RUST_OUTPUT_DIR.absolute()}"
    verification_input = f"PYTHON_CODEBASE_PATH: {python_folder.absolute()}\nRUST_CODEBASE_PATH: {RUST_OUTPUT_DIR.absolute()}"
    code_issues_input = f"PYTHON_CODEBASE_PATH: {python_folder.absolute()}\nRUST_CODEBASE_PATH: {RUST_OUTPUT_DIR.absolute()}"
    
    # Step 1-4: Run Python, Rust, Verification, and Code analyzers in parallel
    log("Steps 1-4: Running Python, Rust, Verification, and Code analyzers in parallel...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all four analyzer tasks
        python_future = executor.submit(run_agent, PYTHON_ANALYZER_PROMPT, python_input)
        rust_future = executor.submit(run_agent, RUST_ANALYZER_PROMPT, rust_input)
        verification_future = executor.submit(run_agent, VERIFICATION_ANALYZER_PROMPT, verification_input)
        code_issues_future = executor.submit(run_agent, CODE_ANALYZER_PROMPT, code_issues_input)
        
        # Wait for all to complete and get results
        python_analysis = python_future.result()
        log("Python codebase analysis completed")
        
        rust_analysis = rust_future.result()
        log("Rust codebase analysis completed")
        
        verification_analysis = verification_future.result()
        log("PyO3 verification analysis completed")
        
        code_issues_analysis = code_issues_future.result()
        log("Code issues analysis completed")
    
    log("All four analyzers completed successfully")
    
    # Step 5: Synthesize analyses into unified blueprint
    log("Step 5: Synthesizing analyses into unified blueprint...")
    synthesis_input = f"PYTHON_ANALYSIS:\n{python_analysis}\n\nRUST_ANALYSIS:\n{rust_analysis}\n\nVERIFICATION_ANALYSIS:\n{verification_analysis}\n\nCODE_ISSUES_ANALYSIS:\n{code_issues_analysis}"
    blueprint = run_agent(ARCHITECT_PROMPT, synthesis_input)
    log("Architectural blueprint synthesis completed")
    
    # Step 6: Run tests to identify current failures
    log("Step 6: Running verification tests to identify current failures...")
    test_runner_input = f"RUST_CODEBASE_PATH: {RUST_OUTPUT_DIR.absolute()}"
    test_runner_report = run_agent(TEST_RUNNER_PROMPT, test_runner_input)
    log("Test runner analysis completed")
    
    return blueprint, test_runner_report


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


def process_module(module_name: str, architectural_blueprint: str, python_folder: Path, test_runner_report: str):
    log(f"\n=== Processing Module: {module_name} ===")
    log(f"Working on module capability: {module_name}")

    # Planner will read files directly as needed

    # Include architectural blueprint, target module, and test runner findings in planner input
    planner_input = f"ARCHITECTURAL_BLUEPRINT:\n{architectural_blueprint}\n\nTARGET_MODULE:\n{module_name}\n\nPYTHON_CODEBASE_PATH:\n{python_folder}\n\nRUST_CODEBASE_PATH:\n{RUST_OUTPUT_DIR}\n\nTEST_RUNNER_FINDINGS:\n{test_runner_report}"
    
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
        # Clean up target directories at the start of each capability
        cleanup_target_directories()
        
        # Extract capability name from the numbered line (e.g. "1. Choice System" -> "Choice System")
        capability_name = current_capability.split('.', 1)[1].strip() if '.' in current_capability else current_capability.strip()
        
        log(f"\n--- Processing Capability {capability_num}/{len(todo_capabilities)} ---")
        log(f"Current capability: {current_capability}")

        # Create context for this specific capability
        capability_context = f"ARCHITECTURAL_BLUEPRINT:\n{architectural_blueprint}\n\nTARGET_MODULE:\n{module_name}\n\nCURRENT_CAPABILITY:\n{current_capability}\n\nPYTHON_CODEBASE_PATH:\n{python_folder}\n\nRUST_CODEBASE_PATH:\n{RUST_OUTPUT_DIR}"

        capability_iteration = 0
        while True:
            capability_iteration += 1
            log(f"{capability_name} - {capability_num}/{len(todo_capabilities)} (Iteration {capability_iteration})")

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

                # Run PyO3 verification to validate Rust against Python implementation
                pyo3_result = run_pyo3_verification()
                
                qa_input = f"CAPABILITY:\n{current_capability}\n\nCODE CHANGES:\n{coder_out}\n\nVERIFICATION:\n{verifier_out}\n\nPYO3_VERIFICATION:\n{pyo3_result}\n\nOVERALL PLAN:\n{plan}"
                qa_out = run_agent(QA_PROMPT, qa_input)
                
                # Extract the actual decision from QA's potentially verbose response
                decision = run_agent(DECISION_EXTRACTOR_PROMPT, qa_out)
                log(f"QA decision extracted: {decision}")
            except Exception as e:
                log(f"Error during verification/QA: {e}")
                capability_context += f"\n\nERROR FEEDBACK:\nVerification/QA failed: {e}"
                continue

            if decision.strip() == "OK":
                log(f"✓ QA declares entire module complete. Finishing module processing.")
                try:
                    # Run documentation agent before commit
                    log(f"Running documentation agent for completed module: {module_name}")
                    documentation_context = f"MODULE: {module_name}\nCOMPLETED_WORK: Entire module implementation\nRUST_OUTPUT_DIR: {RUST_OUTPUT_DIR}"
                    doc_summary = run_agent(DOCUMENTATION_PROMPT, documentation_context)
                    log(f"DOCUMENTATION SUMMARY: {doc_summary}")
                    
                    commit_msg = run_agent(COMMIT_PROMPT, f"Completed entire module: {module_name}")
                    log(f"COMMIT MESSAGE: {commit_msg}")
                except Exception as e:
                    log(f"Warning: Commit failed: {e}")
                return  # Exit the entire module processing
            elif decision.strip() == "NEXT":
                log(f"✓ Capability {capability_num} completed. QA requests moving to next capability.")
                try:
                    # Run documentation agent before commit
                    log(f"Running documentation agent for completed capability: {capability_name}")
                    documentation_context = f"MODULE: {module_name}\nCAPABILITY: {current_capability}\nCOMPLETED_WORK: Single capability implementation\nRUST_OUTPUT_DIR: {RUST_OUTPUT_DIR}"
                    doc_summary = run_agent(DOCUMENTATION_PROMPT, documentation_context)
                    log(f"DOCUMENTATION SUMMARY: {doc_summary}")
                    
                    commit_msg = run_agent(COMMIT_PROMPT, f"Completed capability: {current_capability}")
                    log(f"COMMIT MESSAGE: {commit_msg}")
                except Exception as e:
                    log(f"Warning: Commit failed: {e}")
                break  # Exit current capability loop, continue to next capability
            elif decision.strip().startswith("FAULT:"):
                fault_description = decision.strip()[6:].strip()  # Remove "FAULT:" prefix
                log(f"✗ Capability {capability_num} needs revision: {fault_description}")
                # Update capability context with QA feedback for next iteration
                capability_context += f"\n\nQA FEEDBACK:\n{fault_description}"
            else:
                log(f"✗ Capability {capability_num} unexpected decision: {decision}")
                log(f"Original QA response: {qa_out}")
                # Treat unexpected responses as feedback
                capability_context += f"\n\nQA FEEDBACK:\n{qa_out}"

        log(f"✓ Completed capability {capability_num}: {current_capability.split('.', 1)[1].strip() if '.' in current_capability else current_capability}")


def analyze_existing_log():
    """Analyze existing log to understand what work has been completed"""
    if not LOG_FILE.exists():
        log("No existing log file found, starting fresh.")
        return None
    
    log("Analyzing existing log to understand previous work...")
    try:
        # Check if log file is empty without reading entire contents
        if LOG_FILE.stat().st_size == 0:
            log("Log file is empty, starting fresh.")
            return None
            
        # LogAnalyzer will read the file itself - don't pass contents
        analysis = run_agent(LOG_ANALYZER_PROMPT, "")
        log("=== LOG ANALYSIS ===")
        log(analysis)
        log("=== END LOG ANALYSIS ===")
        
        # Write the analysis back to the log file to replace the raw log
        log("Writing LogAnalyzer output back to log file...")
        try:
            with LOG_FILE.open("w", encoding="utf-8") as f:
                f.write(analysis)
            log("Log file successfully updated with LogAnalyzer output")
        except Exception as e:
            log(f"Warning: Failed to write analysis back to log file: {e}")
        
        return analysis
    except Exception as e:
        log(f"Warning: Log analysis failed: {e}")
        return None


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

    # Use default signal handling - no custom graceful shutdown

    ensure_directories()
    
    # Analyze existing log before starting new work
    log_analysis = analyze_existing_log()
    
    # Continue with existing log after analysis
    log("Orchestrator started.")

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    # Create architectural blueprint and run initial tests
    try:
        architectural_blueprint, test_runner_report = create_architectural_blueprint(folder, log_analysis)
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
    
    # Agents will read Python files directly as needed
    
    # Present plan to user for approval
    print("\n=== EXECUTION PLAN ===")
    print(f"The following {len(modules)} modules will be processed:")
    for i, module in enumerate(modules, 1):
        print(f"  {i}. {module}")
    print()
    
    while True:
        response = input("Do you want to proceed with this plan? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            log("User approved the execution plan")
            break
        elif response in ['n', 'no']:
            log("User declined the execution plan, launching interactive Claude instance")
            print("\nLaunching interactive Claude instance to modify the plan...")
            
            # Launch interactive Claude instance (without -p flag)
            interactive_prompt = (
                "The user has declined the automatically generated execution plan. "
                "Work with them interactively to understand what they want to change about the plan. "
                "The current plan was to process these modules:\n\n"
            )
            for i, module in enumerate(modules, 1):
                interactive_prompt += f"  {i}. {module}\n"
            interactive_prompt += (
                "\nPlease work with the user to understand their concerns and help them create "
                "a modified plan that better suits their needs."
            )
            
            try:
                # Run interactive Claude without -p flag
                subprocess.run(
                    ["claude", "--dangerously-skip-permissions"],
                    input=interactive_prompt,
                    text=True,
                    cwd=RUST_OUTPUT_DIR
                )
            except Exception as e:
                log(f"Error launching interactive Claude: {e}")
                print(f"Error launching interactive Claude: {e}")
            
            # Exit the script after interactive session
            print("Interactive session completed. Please restart the hypervisor with any changes.")
            sys.exit(0)
        else:
            print("Please answer 'y' for yes or 'n' for no.")
    
    # Process each derived module
    for module_name in modules:
        try:
            # Clean up target directories at the start of each module
            cleanup_target_directories()
            
            process_module(module_name, architectural_blueprint, folder, test_runner_report)
        except Exception as e:
            log(f"Error processing module {module_name}: {e}")
            log("Continuing with next module...")

    log("All modules processed.\n")
    summarise_log()

if __name__ == "__main__":
    main()
