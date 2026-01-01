"""
Default Python Server for TechRenaissance Framework
@version 3.6.12
@date 2026-01-01
@author 技术觉醒团队

This is the default Python server that PythonSession connects to.
It provides common tensor operations for development and debugging.

Supported methods:
- add: Element-wise addition of two tensors
- print_tensor: Print tensor in PyTorch format to text file
"""

import sys
from pathlib import Path
from server import RenaissanceServer
import torch


class DefaultPythonServer(RenaissanceServer):
    """
    Default server implementation with common tensor operations.

    Supported methods:
    - add: Add two tensors element-wise
    - print_tensor: Print a tensor using PyTorch's format
    """

    def main_logic(self, method: str, parameters: dict) -> tuple:
        """
        Dispatch method to appropriate handler.

        Args:
            method: Method name ('add', 'print_tensor')
            parameters: Method parameters (optional dict)

        Returns:
            tuple: (success: bool, message: str, result: dict)
                  message indicates output type: "tsr" or "txt"
        """
        if method == "add":
            return self._handle_add()
        elif method == "print_tensor":
            return self._handle_print_tensor()
        else:
            return False, f"Unknown method: {method}", {}

    def _handle_add(self) -> tuple:
        """
        Handle tensor addition: load two input tensors, add them, save result.

        Returns:
            tuple: (success, message="tsr", result={"output_count": 1})
        """
        try:
            # Load input tensors
            tensor_a = self.load_input_tensor(0)
            tensor_b = self.load_input_tensor(1)

            # Perform addition with PyTorch
            result_tensor = tensor_a + tensor_b

            # Save output tensor
            self.save_output_tensor(result_tensor, 0)

            return True, "tsr", {
                "output_count": 1
            }

        except Exception as e:
            return False, f"tsr", {}

    def _handle_print_tensor(self) -> tuple:
        """
        Handle tensor printing: load input tensor, print it to text file.

        The output format is identical to PyTorch's console output.
        Uses stdout redirection to capture exact print() format.

        Returns:
            tuple: (success, message="txt", result={"output_count": 1})
        """
        try:
            # Load input tensor (only the first one)
            tensor = self.load_input_tensor(0)

            # Redirect stdout to file and print tensor
            # This ensures output format matches console exactly
            output_path = self.session_dir / "output_0.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    print(tensor)
                finally:
                    sys.stdout = old_stdout

            return True, "txt", {
                "output_count": 1
            }

        except Exception as e:
            return False, f"txt", {}


# Main entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python default_python_server.py <session_dir>")
        sys.exit(1)

    server = DefaultPythonServer(sys.argv[1])
    server.run()

    # Give filesystem time to close file handles
    import time
    time.sleep(0.1)
