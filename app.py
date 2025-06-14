import os
import subprocess
import tempfile
import sys
from typing import Dict, List, Any, Optional
import json
import traceback
from pathlib import Path
import ast
import re

import autogen
from autogen import ConversableAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
class GeminiConfig:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def generate_response(self, prompt: str, max_tokens: int = 4000) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

class CustomGeminiAgent(ConversableAgent):
    """Custom agent that uses Gemini Flash 2.0 instead of OpenAI"""
    
    def __init__(self, name: str, gemini_config: GeminiConfig, system_message: str, **kwargs):
        # Store system message before calling super().__init__
        self._custom_system_message = system_message
        self.gemini_config = gemini_config
        
        # Initialize parent class with system_message
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=False,  # Disable default LLM
            **kwargs
        )
    
    def generate_reply(self, messages: List[Dict], sender: ConversableAgent, **kwargs) -> str:
        """Generate reply using Gemini Flash 2.0"""
        try:
            # Format conversation history
            conversation = f"System: {self._custom_system_message}\n\n"
            for msg in messages[-5:]:  # Keep last 5 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation += f"{role.title()}: {content}\n"
            
            response = self.gemini_config.generate_response(conversation)
            return response
        except Exception as e:
            return f"Error in generating reply: {str(e)}"

class CodeExecutor:
    """Enhanced code executor with safety checks and output capture"""
    
    def __init__(self, work_dir: Optional[str] = None):
        # Use current directory instead of temp directory
        self.work_dir = work_dir or os.getcwd()
        Path(self.work_dir).mkdir(exist_ok=True)
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code and return results with error handling"""
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0
        }
        
        try:
            if language.lower() == "python":
                result = self._execute_python(code)
            elif language.lower() in ["bash", "shell", "sh"]:
                result = self._execute_bash(code)
            elif language.lower() in ["javascript", "js"]:
                result = self._execute_javascript(code)
            elif language.lower() in ["java"]:
                result = self._execute_java(code)
            elif language.lower() in ["cpp", "c++"]:
                result = self._execute_cpp(code)
            elif language.lower() in ["c"]:
                result = self._execute_c(code)
            elif language.lower() in ["go"]:
                result = self._execute_go(code)
            elif language.lower() in ["rust", "rs"]:
                result = self._execute_rust(code)
            elif language.lower() in ["php"]:
                result = self._execute_php(code)
            elif language.lower() in ["ruby", "rb"]:
                result = self._execute_ruby(code)
            elif language.lower() in ["typescript", "ts"]:
                result = self._execute_typescript(code)
            else:
                result["error"] = f"Unsupported language: {language}"
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely"""
        import time
        import os
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        # Create a permanent file with .py extension and timestamp in current directory
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"python_code_{timestamp}.py"
        
        try:
            # Write code to permanent file with UTF-8 encoding
            with open(code_file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(code)
                f.flush()
                os.fsync(f.fileno())
            
            result["file_path"] = str(code_file)
            
            if not code_file.exists() or code_file.stat().st_size == 0:
                result["error"] = "Failed to create Python file"
                return result
            
            start_time = time.time()
            process = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ Python code saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "Python execution timed out (30 seconds)"
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_javascript(self, code: str) -> Dict[str, Any]:
        """Execute JavaScript code using Node.js"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"javascript_code_{timestamp}.js"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            start_time = time.time()
            process = subprocess.run(
                ["node", str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ JavaScript code saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "JavaScript execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "Node.js not found. Please install Node.js to run JavaScript code."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_java(self, code: str) -> Dict[str, Any]:
        """Execute Java code"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        class_match = re.search(r'public\s+class\s+(\w+)', code)
        class_name = class_match.group(1) if class_match else "Main"
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"{class_name}_{timestamp}.java"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            # Compile
            compile_process = subprocess.run(
                ["javac", str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            
            if compile_process.returncode != 0:
                result["error"] = f"Compilation error: {compile_process.stderr}"
                return result
            
            # Run
            start_time = time.time()
            process = subprocess.run(
                ["java", f"{class_name}_{timestamp}"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ Java code saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "Java execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "Java compiler/runtime not found. Please install Java JDK."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_cpp(self, code: str) -> Dict[str, Any]:
        """Execute C++ code"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"cpp_code_{timestamp}.cpp"
        executable_file = Path(self.work_dir) / f"cpp_executable_{timestamp}"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            # Compile
            compile_process = subprocess.run(
                ["g++", "-o", str(executable_file), str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            
            if compile_process.returncode != 0:
                result["error"] = f"Compilation error: {compile_process.stderr}"
                return result
            
            # Run
            start_time = time.time()
            process = subprocess.run(
                [str(executable_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ C++ code saved to: {code_file}")
            print(f"✅ C++ executable saved to: {executable_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "C++ execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "C++ compiler not found. Please install g++."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_c(self, code: str) -> Dict[str, Any]:
        """Execute C code"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"c_code_{timestamp}.c"
        executable_file = Path(self.work_dir) / f"c_executable_{timestamp}"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            # Compile
            compile_process = subprocess.run(
                ["gcc", "-o", str(executable_file), str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            
            if compile_process.returncode != 0:
                result["error"] = f"Compilation error: {compile_process.stderr}"
                return result
            
            # Run
            start_time = time.time()
            process = subprocess.run(
                [str(executable_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ C code saved to: {code_file}")
            print(f"✅ C executable saved to: {executable_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "C execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "C compiler not found. Please install gcc."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_go(self, code: str) -> Dict[str, Any]:
        """Execute Go code"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"go_code_{timestamp}.go"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            start_time = time.time()
            process = subprocess.run(
                ["go", "run", str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ Go code saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "Go execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "Go not found. Please install Go to run Go code."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_rust(self, code: str) -> Dict[str, Any]:
        """Execute Rust code"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"rust_code_{timestamp}.rs"
        executable_file = Path(self.work_dir) / f"rust_executable_{timestamp}"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            # Compile
            compile_process = subprocess.run(
                ["rustc", "-o", str(executable_file), str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            
            if compile_process.returncode != 0:
                result["error"] = f"Compilation error: {compile_process.stderr}"
                return result
            
            # Run
            start_time = time.time()
            process = subprocess.run(
                [str(executable_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ Rust code saved to: {code_file}")
            print(f"✅ Rust executable saved to: {executable_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "Rust execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "Rust compiler not found. Please install Rust."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_php(self, code: str) -> Dict[str, Any]:
        """Execute PHP code"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"php_code_{timestamp}.php"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                if not code.strip().startswith('<?php'):
                    f.write('<?php\n')
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            start_time = time.time()
            process = subprocess.run(
                ["php", str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ PHP code saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "PHP execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "PHP not found. Please install PHP to run PHP code."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_ruby(self, code: str) -> Dict[str, Any]:
        """Execute Ruby code"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"ruby_code_{timestamp}.rb"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            start_time = time.time()
            process = subprocess.run(
                ["ruby", str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ Ruby code saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "Ruby execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "Ruby not found. Please install Ruby to run Ruby code."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_typescript(self, code: str) -> Dict[str, Any]:
        """Execute TypeScript code using ts-node"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"typescript_code_{timestamp}.ts"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            start_time = time.time()
            process = subprocess.run(
                ["ts-node", str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ TypeScript code saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "TypeScript execution timed out (30 seconds)"
        except FileNotFoundError:
            result["error"] = "ts-node not found. Please install TypeScript and ts-node to run TypeScript code."
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

    def _execute_bash(self, code: str) -> Dict[str, Any]:
        """Execute bash commands safely"""
        import time
        
        result = {"success": False, "output": "", "error": "", "execution_time": 0, "file_path": ""}
        
        timestamp = int(time.time())
        code_file = Path(self.work_dir) / f"bash_script_{timestamp}.sh"
        
        try:
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\n")
                f.write(code)
            
            result["file_path"] = str(code_file)
            
            # Make executable
            os.chmod(code_file, 0o755)
            
            start_time = time.time()
            process = subprocess.run(
                ["/bin/bash", str(code_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
                cwd=self.work_dir,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            end_time = time.time()
            
            result["execution_time"] = end_time - start_time
            result["output"] = process.stdout
            result["error"] = process.stderr
            result["success"] = process.returncode == 0
            
            print(f"✅ Bash script saved to: {code_file}")
            
        except subprocess.TimeoutExpired:
            result["error"] = "Command execution timed out (30 seconds)"
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
        
        return result

class CodeTester:
    """Advanced code testing with multiple test strategies"""
    
    def __init__(self, executor: CodeExecutor):
        self.executor = executor
    
    def test_code(self, code: str, test_cases: List[Dict] = None) -> Dict[str, Any]:
        """Test code with various strategies"""
        results = {
            "syntax_check": False,
            "execution_test": {},
            "unit_tests": [],
            "integration_tests": [],
            "performance_metrics": {},
            "security_check": {},
            "overall_score": 0
        }
        
        # Syntax check
        results["syntax_check"] = self._check_syntax(code)
        
        # Basic execution test
        results["execution_test"] = self.executor.execute_code(code)
        
        # Run custom test cases if provided
        if test_cases:
            results["unit_tests"] = self._run_test_cases(code, test_cases)
        
        # Performance analysis
        results["performance_metrics"] = self._analyze_performance(code)
        
        # Basic security check
        results["security_check"] = self._security_check(code)
        
        # Calculate overall score
        results["overall_score"] = self._calculate_score(results)
        
        return results
    
    def _check_syntax(self, code: str) -> bool:
        """Check Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _run_test_cases(self, code: str, test_cases: List[Dict]) -> List[Dict]:
        """Run provided test cases"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            test_result = {
                "test_id": i,
                "description": test_case.get("description", f"Test {i}"),
                "passed": False,
                "expected": test_case.get("expected"),
                "actual": None,
                "error": None
            }
            
            try:
                # Create test code
                test_code = f"""
{code}

# Test case
{test_case.get('setup', '')}
result = {test_case.get('call', 'main()')}
print(f"RESULT: {{result}}")
"""
                
                execution_result = self.executor.execute_code(test_code)
                
                if execution_result["success"]:
                    # Extract result from output
                    output_lines = execution_result["output"].strip().split('\n')
                    for line in output_lines:
                        if line.startswith("RESULT: "):
                            actual_result = line[8:]  # Remove "RESULT: "
                            test_result["actual"] = actual_result
                            test_result["passed"] = str(test_case.get("expected")) == actual_result
                            break
                else:
                    test_result["error"] = execution_result["error"]
                
            except Exception as e:
                test_result["error"] = str(e)
            
            results.append(test_result)
        
        return results
    
    def _analyze_performance(self, code: str) -> Dict[str, Any]:
        """Basic performance analysis"""
        metrics = {
            "lines_of_code": len(code.split('\n')),
            "complexity_estimate": self._estimate_complexity(code),
            "memory_efficiency": "unknown",
            "time_complexity": "unknown"
        }
        
        return metrics
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity based on control structures"""
        complexity_score = 0
        
        # Count control structures
        complexity_score += len(re.findall(r'\bfor\b', code))
        complexity_score += len(re.findall(r'\bwhile\b', code))
        complexity_score += len(re.findall(r'\bif\b', code))
        complexity_score += len(re.findall(r'\belif\b', code))
        complexity_score += len(re.findall(r'\btry\b', code))
        
        if complexity_score <= 3:
            return "Low"
        elif complexity_score <= 7:
            return "Medium"
        else:
            return "High"
    
    def _security_check(self, code: str) -> Dict[str, Any]:
        """Basic security checks"""
        issues = []
        
        # Check for potentially dangerous functions (more targeted)
        dangerous_patterns = [
            (r'eval\s*\(', "Direct eval() usage"),
            (r'exec\s*\(', "Direct exec() usage"),
            (r'__import__\s*\(.*["\']os["\']', "Dynamic OS module import"),
            (r'os\.system\s*\(', "OS system command execution"),
            (r'subprocess\.(?:call|run|Popen).*shell\s*=\s*True', "Shell command with shell=True"),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Potentially dangerous pattern: {description}")
        
        # Allow legitimate imports and subprocess usage
        # Only flag actual dangerous usage patterns
        
        return {
            "issues": issues,
            "risk_level": "High" if len(issues) > 2 else "Medium" if issues else "Low"
        }
    
    def _calculate_score(self, results: Dict) -> float:
        """Calculate overall code quality score"""
        score = 0
        max_score = 100
        
        # Syntax check (20 points)
        if results["syntax_check"]:
            score += 20
        
        # Execution test (30 points)
        if results["execution_test"].get("success"):
            score += 30
        
        # Unit tests (30 points)
        if results["unit_tests"]:
            passed_tests = sum(1 for test in results["unit_tests"] if test["passed"])
            total_tests = len(results["unit_tests"])
            score += (passed_tests / total_tests) * 30
        
        # Security (20 points)
        if results["security_check"]["risk_level"] == "Low":
            score += 20
        elif results["security_check"]["risk_level"] == "Medium":
            score += 10
        
        return min(score, max_score)
        
class InteractiveCodingInterface:
    """Interactive interface for the Advanced Coding Agent"""
    
    def __init__(self, agent: 'AdvancedCodingAgent'):
        self.agent = agent
        self.session_history = []
    
    def display_welcome(self):
        """Display welcome message and instructions"""
        print("=" * 80)
        print("🤖 ADVANCED AI CODING AGENT")
        print("=" * 80)
        print("Welcome! I can help you solve coding problems, write code, and test solutions.")
        print("\nCommands:")
        print("  • 'solve' - Solve a coding problem")
        print("  • 'code' - Write code for a specific task")
        print("  • 'review' - Review existing code")
        print("  • 'test' - Generate tests for code")
        print("  • 'history' - Show session history")
        print("  • 'help' - Show this help message")
        print("  • 'quit' or 'exit' - Exit the program")
        print("-" * 80)
    
    def get_problem_input(self) -> Dict[str, Any]:
        """Get problem description and requirements from user"""
        print("\n📝 Problem Description:")
        print("Describe the coding problem or task you want me to solve.")
        print("(Press Enter twice when finished, or type 'cancel' to abort)")
        
        lines = []
        empty_lines = 0
        while True:
            try:
                line = input("> " if not lines else "  ")
                if line.lower().strip() == 'cancel':
                    return None
                
                if line.strip() == "":
                    empty_lines += 1
                    if empty_lines >= 2 and lines:
                        break
                else:
                    empty_lines = 0
                
                lines.append(line)
            except KeyboardInterrupt:
                print("\n❌ Input cancelled.")
                return None
        
        problem_description = "\n".join(lines).strip()
        if not problem_description:
            print("❌ No problem description provided.")
            return None
        
        # Get requirements
        print("\n📋 Requirements (optional):")
        print("List any specific requirements, constraints, or features.")
        print("(One per line, press Enter on empty line when finished)")
        
        requirements = []
        while True:
            try:
                req = input("• ").strip()
                if not req:
                    break
                requirements.append(req)
            except KeyboardInterrupt:
                break
        
        # Get programming language preference
        print("\n💻 Programming Language (default: Python):")
        language = input("Language: ").strip() or "python"
        
        return {
            "description": problem_description,
            "requirements": requirements,
            "language": language.lower()
        }
    
    def get_code_input(self) -> str:
        """Get code from user for review or testing"""
        print("\n📄 Paste your code below:")
        print("(Press Ctrl+D or type '###END###' on a new line when finished)")
        
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "###END###":
                    break
                lines.append(line)
        except EOFError:
            pass
        except KeyboardInterrupt:
            print("\n❌ Input cancelled.")
            return None
        
        return "\n".join(lines).strip()
    
    def display_results(self, results: Dict[str, Any]):
        """Display formatted results"""
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        
        success_icon = "✅" if results.get('success', False) else "❌"
        print(f"Status: {success_icon} {'Success' if results.get('success', False) else 'Failed'}")
        
        if 'test_results' in results:
            score = results['test_results'].get('overall_score', 0)
            print(f"Quality Score: {score:.1f}/100")
        
        print(f"Problem: {results.get('problem', 'N/A')}")
        
        # Display final code
        if 'final_code' in results:
            print(f"\n📋 GENERATED CODE:")
            print("-" * 40)
            print(results['final_code'])
        
        # Display execution results
        if 'execution_result' in results:
            exec_result = results['execution_result']
            if exec_result.get('success'):
                print(f"\n✅ EXECUTION OUTPUT:")
                print("-" * 40)
                print(exec_result.get('output', 'No output'))
            else:
                print(f"\n❌ EXECUTION ERROR:")
                print("-" * 40)
                print(exec_result.get('error', 'Unknown error'))
        
        # Display test results
        if 'test_results' in results and results['test_results'].get('unit_tests'):
            print(f"\n🧪 TEST RESULTS:")
            print("-" * 40)
            for test in results['test_results']['unit_tests']:
                status = "✅" if test.get('passed', False) else "❌"
                print(f"{status} {test.get('description', 'Test')}")
                if not test.get('passed', False) and test.get('error'):
                    print(f"   Error: {test['error']}")
        
        # Display review feedback
        if 'review_feedback' in results:
            print(f"\n👀 REVIEW FEEDBACK:")
            print("-" * 40)
            print(results['review_feedback'])
        
        print("=" * 80)
    
    def handle_solve_command(self):
        """Handle the solve command"""
        problem_data = self.get_problem_input()
        if not problem_data:
            return
        
        print(f"\n🚀 Processing your request...")
        print(f"Problem: {problem_data['description'][:100]}{'...' if len(problem_data['description']) > 100 else ''}")
        
        try:
            results = self.agent.solve_coding_problem(
                problem_data['description'],
                problem_data['requirements']
            )
            
            self.display_results(results)
            self.session_history.append({
                "command": "solve",
                "input": problem_data,
                "results": results,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Error processing request: {str(e)}")
    
    def handle_review_command(self):
        """Handle the review command"""
        code = self.get_code_input()
        if not code:
            return
        
        print(f"\n👀 Reviewing your code...")
        
        try:
            # Use the code reviewer agent directly
            review_prompt = f"""
Please review this code and provide detailed feedback:

```python
{code}
```

Focus on:
1. Code quality and best practices
2. Potential bugs or issues
3. Security vulnerabilities
4. Performance optimizations
5. Readability and maintainability
6. Specific suggestions for improvement
"""
            
            review_feedback = self.agent.agents["code_reviewer"].generate_reply(
                [{"role": "user", "content": review_prompt}], None
            )
            
            print("\n👀 CODE REVIEW:")
            print("-" * 40)
            print(review_feedback)
            
            self.session_history.append({
                "command": "review",
                "input": {"code": code},
                "results": {"review_feedback": review_feedback},
                "timestamp": __import__('datetime').datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Error reviewing code: {str(e)}")
    
    def handle_test_command(self):
        """Handle the test command"""
        code = self.get_code_input()
        if not code:
            return
        
        print(f"\n🧪 Generating tests for your code...")
        
        try:
            # Run comprehensive tests
            test_results = self.agent.tester.test_code(code)
            
            print(f"\n🧪 TEST RESULTS:")
            print("-" * 40)
            print(f"Syntax Check: {'✅' if test_results['syntax_check'] else '❌'}")
            print(f"Execution: {'✅' if test_results['execution_test']['success'] else '❌'}")
            print(f"Overall Score: {test_results['overall_score']:.1f}/100")
            
            if test_results['execution_test'].get('error'):
                print(f"Execution Error: {test_results['execution_test']['error']}")
            
            if test_results['security_check']['issues']:
                print(f"\n⚠️ Security Issues:")
                for issue in test_results['security_check']['issues']:
                    print(f"  • {issue}")
            
            self.session_history.append({
                "command": "test",
                "input": {"code": code},
                "results": test_results,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Error testing code: {str(e)}")
    
    def handle_history_command(self):
        """Handle the history command"""
        if not self.session_history:
            print("📝 No session history available.")
            return
        
        print(f"\n📝 SESSION HISTORY ({len(self.session_history)} items):")
        print("-" * 40)
        
        for i, item in enumerate(self.session_history[-10:], 1):  # Show last 10 items
            timestamp = item['timestamp'][:19].replace('T', ' ')
            command = item['command'].upper()
            
            if command == "SOLVE":
                description = item['input']['description'][:50] + "..." if len(item['input']['description']) > 50 else item['input']['description']
                success = "✅" if item['results'].get('success') else "❌"
                print(f"{i}. [{timestamp}] {command} - {description} {success}")
            else:
                print(f"{i}. [{timestamp}] {command}")
    
    def run(self):
        """Main interactive loop"""
        self.display_welcome()
        
        while True:
            try:
                print()
                command = input("🤖 Enter command (type 'help' for options): ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for using the Advanced AI Coding Agent!")
                    break
                elif command == 'help':
                    self.display_welcome()
                elif command == 'solve':
                    self.handle_solve_command()
                elif command == 'code':
                    self.handle_solve_command()  # Same as solve for now
                elif command == 'review':
                    self.handle_review_command()
                elif command == 'test':
                    self.handle_test_command()
                elif command == 'history':
                    self.handle_history_command()
                elif command == '':
                    continue
                else:
                    print(f"❌ Unknown command: '{command}'. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the Advanced AI Coding Agent!")
                break
            except EOFError:
                print("\n\n👋 Goodbye! Thanks for using the Advanced AI Coding Agent!")
                break


class AdvancedCodingAgent:
    """Main orchestrator for the advanced coding agent system"""
    
    def __init__(self, gemini_api_key: str = None, work_dir: Optional[str] = None):
        # Get API key from parameter or environment
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided either as parameter or in environment variables")
        
        self.gemini_config = GeminiConfig(api_key)
        self.executor = CodeExecutor(work_dir)
        self.tester = CodeTester(self.executor)
        self.agents = self._create_agents()
        self.conversation_history = []
    
    def _create_agents(self) -> Dict[str, ConversableAgent]:
        """Create specialized agents for different tasks"""
        
        try:
            # Code Writer Agent
            code_writer = CustomGeminiAgent(
                name="CodeWriter",
                gemini_config=self.gemini_config,
                system_message="""You are an expert software developer. Your role is to:
1. Write clean, efficient, and well-documented code
2. Follow best practices and coding standards
3. Include proper error handling
4. Write modular and reusable code
5. Add helpful comments and docstrings

Always provide complete, working code solutions. Format your code in markdown code blocks with the appropriate language specified.""",
                human_input_mode="NEVER"
            )
            
            # Code Reviewer Agent
            code_reviewer = CustomGeminiAgent(
                name="CodeReviewer",
                gemini_config=self.gemini_config,
                system_message="""You are a senior code reviewer. Your role is to:
1. Review code for bugs, issues, and improvements
2. Check for security vulnerabilities
3. Suggest performance optimizations
4. Ensure code follows best practices
5. Provide constructive feedback

Always provide detailed feedback with specific suggestions for improvement.""",
                human_input_mode="NEVER"
            )
            
            # Test Writer Agent
            test_writer = CustomGeminiAgent(
                name="TestWriter",
                gemini_config=self.gemini_config,
                system_message="""You are a QA engineer specializing in test automation. Your role is to:
1. Create comprehensive test cases
2. Write unit tests, integration tests, and edge case tests
3. Ensure good test coverage
4. Create both positive and negative test scenarios
5. Write clear test documentation

Provide test cases in a structured format that can be executed programmatically.""",
                human_input_mode="NEVER"
            )
            
            return {
                "code_writer": code_writer,
                "code_reviewer": code_reviewer,
                "test_writer": test_writer
            }
            
        except Exception as e:
            print(f"Error creating agents: {str(e)}")
            raise
    
    def solve_coding_problem(self, problem_description: str, requirements: List[str] = None) -> Dict[str, Any]:
        """Main method to solve a coding problem end-to-end"""
        
        print(f"🚀 Starting to solve: {problem_description}")
        
        # Step 1: Generate initial code
        print("\n📝 Step 1: Generating initial code...")
        code_prompt = f"""
Problem: {problem_description}

Requirements:
{chr(10).join(f"- {req}" for req in (requirements or []))}

Please write a complete Python solution for this problem. Include:
1. A main function that demonstrates the solution
2. Proper error handling
3. Clear documentation
4. Example usage
"""
        
        initial_code = self.agents["code_writer"].generate_reply([{"role": "user", "content": code_prompt}], None)
        code_block = self._extract_code_block(initial_code)
        
        if not code_block:
            return {"error": "Failed to generate initial code"}
        
        print(f"✅ Initial code generated ({len(code_block.split(chr(10)))} lines)")
        
        # Step 2: Execute and test initial code
        print("\n🔧 Step 2: Testing initial code...")
        execution_result = self.executor.execute_code(code_block)
        
        if not execution_result["success"]:
            print(f"❌ Initial code failed: {execution_result['error']}")
            # Try to fix the code
            fix_prompt = f"""
The following code has an error:

```python
{code_block}
```

Error: {execution_result['error']}

Please fix this code and provide the corrected version.
"""
            fixed_code = self.agents["code_writer"].generate_reply([{"role": "user", "content": fix_prompt}], None)
            code_block = self._extract_code_block(fixed_code) or code_block
            execution_result = self.executor.execute_code(code_block)
        
        print(f"✅ Code execution: {'Success' if execution_result['success'] else 'Failed'}")
        
        # Step 3: Generate test cases
        print("\n🧪 Step 3: Generating test cases...")
        test_prompt = f"""
For the following code, create comprehensive test cases:

```python
{code_block}
```

Problem: {problem_description}

Create test cases in this JSON format:
[
    {{
        "description": "Test description",
        "setup": "any setup code if needed",
        "call": "function_call_to_test()",
        "expected": "expected_result"
    }}
]

Provide at least 5 different test cases including edge cases.
"""
        
        test_response = self.agents["test_writer"].generate_reply([{"role": "user", "content": test_prompt}], None)
        test_cases = self._extract_test_cases(test_response)
        
        print(f"✅ Generated {len(test_cases)} test cases")
        
        # Step 4: Run comprehensive tests
        print("\n🔍 Step 4: Running comprehensive tests...")
        test_results = self.tester.test_code(code_block, test_cases)
        
        print(f"✅ Testing complete - Overall score: {test_results['overall_score']:.1f}/100")
        
        # Step 5: Code review and optimization
        print("\n👀 Step 5: Code review...")
        review_prompt = f"""
Please review this code and provide suggestions for improvement:

```python
{code_block}
```

Test Results Summary:
- Syntax Check: {'✅' if test_results['syntax_check'] else '❌'}
- Execution: {'✅' if test_results['execution_test']['success'] else '❌'}
- Tests Passed: {sum(1 for t in test_results['unit_tests'] if t['passed'])}/{len(test_results['unit_tests'])}
- Security Risk: {test_results['security_check']['risk_level']}
- Overall Score: {test_results['overall_score']:.1f}/100

Provide specific suggestions for improvement.
"""
        
        review_feedback = self.agents["code_reviewer"].generate_reply([{"role": "user", "content": review_prompt}], None)
        
        print("✅ Code review completed")
        
        # Compile final results
        results = {
            "problem": problem_description,
            "requirements": requirements or [],
            "final_code": code_block,
            "execution_result": execution_result,
            "test_results": test_results,
            "review_feedback": review_feedback,
            "success": execution_result["success"] and test_results["overall_score"] > 60
        }
        
        print(f"\n🎉 Problem solving complete! Success: {'✅' if results['success'] else '❌'}")
        
        return results
    
    def _extract_code_block(self, text: str) -> Optional[str]:
        """Extract Python code block from markdown text"""
        patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'`([^`\n]+)`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code blocks found, return the entire text (might be plain code)
        lines = text.split('\n')
        if any('def ' in line or 'import ' in line or 'class ' in line for line in lines):
            return text.strip()
        
        return None
    
    def _extract_test_cases(self, text: str) -> List[Dict]:
        """Extract test cases from response"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback: create basic test cases
        return [
            {
                "description": "Basic functionality test",
                "setup": "",
                "call": "main() if 'main' in globals() else None",
                "expected": "None"
            }
        ]

# Example usage and demonstration
def main():
    """Interactive main function for the Advanced Coding Agent"""
    
    # Load API key from .env file
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please make sure you have a .env file with:")
        print("GEMINI_API_KEY=your_api_key_here")
        print("\nYou can get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    print("🔑 API key loaded successfully from .env file")
    
    try:
        agent = AdvancedCodingAgent()  # API key will be loaded from .env
        print("✅ Advanced Coding Agent initialized successfully")
        
        # Start interactive interface
        interface = InteractiveCodingInterface(agent)
        interface.run()
        
    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()