"""
Python Server Test Example
@version 3.6.8
@date 2025-12-27
@author 技术觉醒团队

This file demonstrates how to inherit from TechRenaissanceServer
to implement specific tensor computation logic.
"""

import sys
from pathlib import Path

# Add scripts directory to path to import server.py
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from server import RenaissanceServer
import torch


class PythonServerExample(RenaissanceServer):
    """
    Example server implementation that supports tensor addition.

    Supported methods:
    - add: Element-wise addition of two tensors
    """

    def main_logic(self, method: str, parameters: dict) -> tuple:
        """
        Implement specific computation logic.

        Args:
            method: Method name ('add')
            parameters: Method parameters (currently empty for add)

        Returns:
            tuple: (success, message, result)
        """
        if method == "add":
            return self._handle_add()
        else:
            return False, f"Unknown method: {method}", {}

    def _handle_add(self) -> tuple:
        """
        Handle tensor addition: load two input tensors, add them, save result.

        Returns:
            tuple: (success, message, result)
        """
        try:
            # Load input tensors
            tensor_a = self.load_input_tensor(0)
            tensor_b = self.load_input_tensor(1)

            # Perform addition with PyTorch
            result_tensor = tensor_a + tensor_b

            # Save output tensor
            self.save_output_tensor(result_tensor, 0)

            return True, "Addition completed successfully", {
                "output_count": 1
            }

        except Exception as e:
            return False, f"Addition failed: {str(e)}", {}


# For direct testing (can be invoked by server.py's main())
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_python_server.py <session_dir>")
        sys.exit(1)

    server = PythonServerExample(sys.argv[1])
    server.run()

    # 给文件系统一点时间确保所有文件句柄都关闭
    import time
    time.sleep(0.1)
