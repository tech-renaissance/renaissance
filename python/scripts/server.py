"""
Python Server Base Class for Renaissance C++/Python Interoperability
@version 3.6.8
@date 2025-12-27
@author 技术觉醒团队
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import TSR I/O functions
try:
    from tsr_io import import_tensor, export_tensor
except ImportError:
    print("ERROR: Cannot import tsr_io. Make sure tsr_io.py is in the same directory.", file=sys.stderr)
    sys.exit(1)


class RenaissanceServer:
    """
    Base class for Python server that communicates with C++ via file I/O.

    The server expects to find:
    - request.json: Contains 'method' and 'parameters' fields
    - response.json: Should contain 'success', 'message', and 'result' fields

    For tensor operations:
    - Input tensors are in input_0.tsr, input_1.tsr, ...
    - Output tensors should be written to output_0.tsr, output_1.tsr, ...
    """

    def __init__(self, session_dir: str):
        """
        Initialize the server with a session directory.

        Args:
            session_dir: Path to the session directory containing request.json
        """
        self.session_dir = Path(session_dir)
        self.request_path = self.session_dir / "request.json"
        self.response_path = self.session_dir / "response.json"

        if not self.request_path.exists():
            self._write_error_response(f"request.json not found in {session_dir}")

    def run(self):
        """
        Main entry point: read request, process command, write response.

        This method:
        1. Reads request.json
        2. Calls main_logic() to process the command
        3. Writes the result to response.json
        """
        try:
            # Read request
            with open(self.request_path, 'r', encoding='utf-8') as f:
                request = json.load(f)

            method = request.get('method')
            parameters = request.get('parameters', {})

            if method is None:
                self._write_error_response("Request missing 'method' field")
                return

            # Call the main logic (to be implemented by subclasses)
            success, message, result = self.main_logic(method, parameters)

            # Write response
            response = {
                'success': success,
                'message': message,
                'result': result
            }

            with open(self.response_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2)

        except Exception as e:
            self._write_error_response(f"Server error: {str(e)}")

    def main_logic(self, method: str, parameters: Dict) -> tuple:
        """
        Process the command and return result.

        This method should be overridden by subclasses to implement
        specific computation logic.

        Args:
            method: The method name to execute (e.g., 'add', 'matmul')
            parameters: Dictionary of parameters for the method

        Returns:
            tuple: (success: bool, message: str, result: dict)
                - success: True if operation succeeded
                - message: Description of the result or error
                - result: Dictionary containing return values (e.g., output tensor paths)
        """
        raise NotImplementedError("Subclasses must implement main_logic()")

    def load_input_tensor(self, index: int):
        """
        Load an input tensor from TSR V3 format.

        Args:
            index: Input tensor index (0 for input_0.tsr, 1 for input_1.tsr, ...)

        Returns:
            PyTorch tensor
        """
        tensor_path = self.session_dir / f"input_{index}.tsr"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Input tensor file not found: {tensor_path}")

        return import_tensor(str(tensor_path))

    def save_output_tensor(self, tensor, index: int):
        """
        Save an output tensor to TSR V3 format.

        Args:
            tensor: PyTorch tensor to save
            index: Output tensor index (0 for output_0.tsr, 1 for output_1.tsr, ...)
        """
        tensor_path = self.session_dir / f"output_{index}.tsr"
        export_tensor(tensor, str(tensor_path))

    def _write_error_response(self, message: str):
        """
        Write an error response to response.json.

        Args:
            message: Error message
        """
        response = {
            'success': False,
            'message': message,
            'result': {}
        }

        with open(self.response_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2)


def main():
    """
    Main entry point when invoked by C++.

    Usage:
        python server.py <session_dir>
    """
    if len(sys.argv) != 2:
        print("Usage: python server.py <session_dir>", file=sys.stderr)
        sys.exit(1)

    session_dir = sys.argv[1]

    # 导入实际的服务器实现
    # 注意：需要将tests目录添加到Python路径
    tests_dir = Path(__file__).parent.parent / "tests"
    sys.path.insert(0, str(tests_dir))

    try:
        from test_python_server import PythonServerExample
        server = PythonServerExample(session_dir)
        server.run()
    except ImportError as e:
        print(f"ERROR: Cannot import PythonServerExample: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Server execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
