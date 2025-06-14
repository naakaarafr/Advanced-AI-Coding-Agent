# Advanced AI Coding Agent

A powerful AI-powered coding assistant that can solve programming problems, review code, generate tests, and execute code across multiple programming languages. Built with Google's Gemini 2.0 Flash model and AutoGen framework.

## 🚀 Features

### Core Capabilities
- **Problem Solving**: Describe a coding problem and get a complete solution
- **Code Review**: Get detailed feedback on code quality, security, and best practices
- **Test Generation**: Automatically generate comprehensive test cases
- **Multi-language Support**: Execute code in 12+ programming languages
- **Interactive Interface**: User-friendly command-line interface
- **Code Quality Analysis**: Comprehensive scoring system for code quality

### Supported Languages
- Python
- JavaScript/Node.js
- TypeScript
- Java
- C/C++
- Go
- Rust
- PHP
- Ruby
- Bash/Shell
- And more...

### AI Agents
The system uses specialized AI agents for different tasks:
- **CodeWriter**: Generates clean, efficient code solutions
- **CodeReviewer**: Provides detailed code reviews and suggestions
- **TestWriter**: Creates comprehensive test cases

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Node.js (for JavaScript/TypeScript execution)
- Java JDK (for Java execution)
- GCC/G++ (for C/C++ execution)
- Go, Rust, PHP, Ruby (optional, for respective language support)

### Python Dependencies
```bash
pip install autogen-agentchat google-generativeai python-dotenv
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/naakaarafr/Advanced-AI-Coding-Agent
   cd Advanced-AI-Coding-Agent
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

4. **Get your Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Generate a new API key
   - Add it to your `.env` file

## 🚀 Usage

### Interactive Mode
Run the main application to start the interactive interface:
```bash
python app.py
```

### Available Commands
- `solve` - Solve a coding problem
- `code` - Write code for a specific task
- `review` - Review existing code
- `test` - Generate and run tests for code
- `history` - Show session history
- `help` - Show available commands
- `quit` or `exit` - Exit the application

### Example Usage

#### Solving a Problem
```
🤖 Enter command: solve

📝 Problem Description:
> Create a function that finds the two numbers in an array that add up to a target sum
> The function should return the indices of these two numbers
> 

📋 Requirements (optional):
• Should handle edge cases
• Should be efficient
• Should work with negative numbers
• 

💻 Programming Language (default: Python):
Language: python

🚀 Processing your request...
```

#### Code Review
```
🤖 Enter command: review

📄 Paste your code below:
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)
###END###

👀 Reviewing your code...
```

## 🏗️ Architecture

### Core Components

#### 1. AdvancedCodingAgent
Main orchestrator that coordinates all system components.

#### 2. CustomGeminiAgent
Custom agent implementation using Google's Gemini 2.0 Flash model.

#### 3. CodeExecutor
Handles code execution across multiple programming languages with safety checks.

#### 4. CodeTester
Comprehensive testing framework with multiple test strategies:
- Syntax validation
- Execution testing
- Unit test generation
- Performance analysis
- Security checks

#### 5. InteractiveCodingInterface
User-friendly command-line interface for interaction.

### Workflow
1. **Problem Input**: User describes the coding problem
2. **Code Generation**: AI generates initial solution
3. **Code Execution**: System tests the generated code
4. **Test Generation**: AI creates comprehensive test cases
5. **Quality Analysis**: System analyzes code quality and security
6. **Code Review**: AI provides detailed feedback and suggestions
7. **Results**: User receives complete solution with analysis

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Customization
You can customize the system by modifying:
- Agent system messages in `_create_agents()`
- Code execution timeouts in `CodeExecutor`
- Security checks in `_security_check()`
- Quality scoring in `_calculate_score()`

## 📊 Code Quality Metrics

The system provides comprehensive code quality analysis:

### Scoring System (0-100)
- **Syntax Check**: 20 points
- **Execution Success**: 30 points
- **Test Coverage**: 30 points
- **Security**: 20 points

### Analysis Areas
- Code complexity estimation
- Security vulnerability detection
- Performance considerations
- Best practices compliance
- Error handling quality

## 🔒 Security Features

### Safety Measures
- Sandboxed code execution
- Dangerous pattern detection
- Timeout protection (30 seconds)
- Limited file system access
- Security risk assessment

### Detected Patterns
- Direct `eval()` usage
- `exec()` calls
- OS system commands
- Shell injection risks
- Dynamic imports

## 📁 File Structure

```
advanced-coding-agent/
├── app.py                 # Main application file
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── generated_files/      # Generated code files (created automatically)
    ├── python_code_*.py
    ├── javascript_code_*.js
    ├── java_code_*.java
    └── ...
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions
- Include type hints where appropriate

## 📝 Examples

### Example 1: Fibonacci Sequence
```python
# User input: "Create a function to generate Fibonacci sequence"
def fibonacci(n):
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence
```

### Example 2: Binary Search
```python
# User input: "Implement binary search algorithm"
def binary_search(arr, target):
    """Binary search implementation."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Error
```
❌ Error: GEMINI_API_KEY not found in environment variables
```
**Solution**: Ensure your `.env` file contains a valid Gemini API key.

#### 2. Language Not Found
```
❌ Node.js not found. Please install Node.js to run JavaScript code.
```
**Solution**: Install the required language runtime on your system.

#### 3. Code Execution Timeout
```
❌ Execution timed out (30 seconds)
```
**Solution**: Optimize your code or check for infinite loops.

### Getting Help
- Check the console output for detailed error messages
- Use the `help` command for available options
- Review the generated code files for debugging

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini 2.0 Flash for AI capabilities
- AutoGen framework for agent orchestration
- The open-source community for inspiration and tools

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Review the examples
3. Open an issue on GitHub
4. Check the documentation

---

**Happy Coding! 🚀**
