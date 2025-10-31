from comfy_api.latest import io

class BenchmarkWorkflow(io.ComfyNode):
    """
    A node to control benchmarking in comfyui-benchmark extension.
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
                    "file_prefix",
                    default="",
                    multiline=False
                ),
                io.String.Input(
                    "file_postfix",
                    default="",
                    multiline=False
                )
            ],
            outputs=[]
        )

    @classmethod
    def execute(cls, capture_benchmark, file_prefix, file_postfix) -> io.NodeOutput:
        return io.NodeOutput()
