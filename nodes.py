from typing_extensions import override
from comfy_api.latest import io

class BenchmarkWorkflow(io.ComfyNode):
    """
    A node to control benchmarking in comfyui-benchmark extension.

    Class methods
    ------------
    define_schema (io.Schema):
        Defines the metadata, input, and output parameters of the node.
    execute:
        Processes the node inputs and returns outputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Defines the schema for the BenchmarkWorkflow node, specifying metadata and parameters.
        """
        return io.Schema(
            node_id="BenchmarkWorkflow",
            display_name="Benchmark Workflow",
            category="_for_testing/benchmark",
            inputs=[
                io.Boolean.Input(
                    "capture_benchmark",
                    default=True
                ),
                io.String.Input(
                    "outfile_postfix1",
                    default="",
                    multiline=False
                ),
                io.String.Input(
                    "outfile_postfix2",
                    default="",
                    multiline=False
                )
            ],
            outputs=[
                io.Boolean.Output(),
                io.String.Output(),
                io.String.Output()
            ]
        )

    @classmethod
    def execute(cls, capture_benchmark, outfile_postfix1, outfile_postfix2) -> io.NodeOutput:
        """
        Executes the node, returning the input values as outputs.

        Args:
            capture_benchmark (bool): Whether to enable benchmarking for the workflow.
            outfile_postfix1 (str): First postfix for the benchmark output file.
            outfile_postfix2 (str): Second postfix for the benchmark output file.

        Returns:
            io.NodeOutput: Contains the capture_benchmark boolean, outfile_postfix1 string, and outfile_postfix2 string.
        """
        return io.NodeOutput(capture_benchmark, outfile_postfix1, outfile_postfix2)