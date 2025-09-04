# NOTE: this file was 99% written by Claude Code
import json
import sys
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_nvidia_smi_line(line, workflow_start_datetime, workflow_start_time):
    parts = line.strip().split(', ')
    if len(parts) < 12:
        return None

    timestamp_dt = datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f')
    # Convert to relative seconds from workflow start
    time_delta = timestamp_dt - workflow_start_datetime
    relative_time = time_delta.total_seconds()

    return {
        'timestamp': timestamp_dt,
        'relative_time': relative_time,
        'memory_used': int(parts[1]),
        'memory_total': int(parts[2]),
        'gpu_utilization': int(parts[3]),
        'memory_utilization': int(parts[4]),
        'power_draw': float(parts[5]),
        'power_instant': float(parts[6]),
        'power_limit': float(parts[7]),
    }


def create_benchmark_visualization(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    device_name = data['device_info']['name'].split(' ')[1] + ' ' + data['device_info']['name'].split(' ')[2] + ' ' + data['device_info']['name'].split(' ')[3] + ' ' + data['device_info']['name'].split(' ')[4]
    total_vram = data['device_info']['total_vram']

    workflow_start = data['benchmark_data']['workflow_start_time']
    workflow_end = data['benchmark_data']['workflow_end_time']
    workflow_start_datetime = datetime.strptime(data['benchmark_data']['workflow_start_datetime'], '%Y/%m/%d %H:%M:%S.%f')

    # Parse nvidia-smi data if it exists
    nvidia_smi_data = []
    if 'nvidia_smi_data' in data and data['nvidia_smi_data']:
        for line in data['nvidia_smi_data']:
            parsed = parse_nvidia_smi_line(line, workflow_start_datetime, workflow_start)
            if parsed:
                nvidia_smi_data.append(parsed)

    # Determine which graphs to show based on available data
    has_nvidia_data = len(nvidia_smi_data) > 0

    if has_nvidia_data:
        # Use relative times starting from 0
        relative_times = [d['relative_time'] for d in nvidia_smi_data]
        memory_used = [d['memory_used'] for d in nvidia_smi_data]
        gpu_utilization = [d['gpu_utilization'] for d in nvidia_smi_data]
        power_draw = [d['power_draw'] for d in nvidia_smi_data]
        power_limit = nvidia_smi_data[0]['power_limit'] if nvidia_smi_data else None

        # Create subplot with all graphs
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'VRAM Usage - {device_name}', 'GPU Utilization', 'Power Usage', 'Workflow Operations Timeline'),
            row_heights=[0.35, 0.2, 0.2, 0.25]
        )
        operations_row = 4
    else:
        # Create subplot with only operations timeline
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Workflow Operations Timeline',),
            row_heights=[1.0]
        )
        operations_row = 1

    if has_nvidia_data:
        # Add VRAM Usage
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=memory_used,
                name='VRAM Used (MB)',
                mode='lines',
                line=dict(color='darkblue', width=2),
                fill='tozeroy',
                hovertemplate='<b>Time</b>: %{x:.2f}s<br>' +
                              '<b>VRAM</b>: %{y} MB<br>' +
                              '<b>Percentage</b>: %{customdata:.1f}%<extra></extra>',
                customdata=[(m/total_vram)*100 for m in memory_used]
            ),
            row=1, col=1
        )

        fig.add_hline(y=total_vram, line_dash="dash", line_color="red", 
                      annotation_text=f"Max VRAM: {total_vram} MB",
                      annotation_position="top right",
                      row=1, col=1)

        # Add GPU Utilization
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=gpu_utilization,
                name='GPU Utilization (%)',
                mode='lines',
                line=dict(color='orange', width=2),
                fill='tozeroy',
                hovertemplate='<b>Time</b>: %{x:.2f}s<br>' +
                              '<b>GPU Utilization</b>: %{y}%<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                      annotation_text="100%",
                      annotation_position="top right",
                      row=2, col=1)

        # Add Power Usage
        power_percentages = [(p/power_limit)*100 if power_limit else 0 for p in power_draw]
        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=power_draw,
                name='Power Draw (W)',
                mode='lines',
                line=dict(color='green', width=2),
                fill='tozeroy',
                hovertemplate='<b>Time</b>: %{x:.2f}s<br>' +
                              '<b>Power Draw</b>: %{y:.1f}W<br>' +
                              '<b>Power Limit %</b>: %{customdata:.1f}%<extra></extra>',
                customdata=power_percentages
            ),
            row=3, col=1
        )

        if power_limit:
            fig.add_hline(y=power_limit, line_dash="dash", line_color="red", 
                          annotation_text=f"Power Limit: {power_limit:.0f}W",
                          annotation_position="top right",
                          row=3, col=1)

    operations = []
    colors = {
        'load_torch_file': 'purple',
        'model_load': 'orange',
        'load_state_dict': 'brown',
        'load_diffusion_model': 'indigo',
        'sampling': 'green',
        'sampler_sample': 'cyan',
        'vae_encode': 'blue',
        'vae_decode': 'red',
        'clip_tokenize': 'magenta',
        'clip_encode': 'pink'
    }

    # Handle both old and new data format
    if 'load_data' in data:
        # New format - load_data dictionary
        for item in data['load_data'].get('load_torch_file', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                model_name = item['ckpt'].split('\\')[-1].split('.')[0]
                operations.append({
                    'type': 'load_torch_file',
                    'name': f'Load: {model_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time']
                })

        for item in data['load_data'].get('model_load', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                operations.append({
                    'type': 'model_load',
                    'name': f'Model Load: {item["model"]}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time']
                })

        # New load_state_dict operations
        for item in data['load_data'].get('load_state_dict', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                func_name = item.get('func_name', 'load_state_dict')
                operations.append({
                    'type': 'load_state_dict',
                    'name': f'Load State Dict: {func_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time'],
                    'func_name': func_name
                })

        # New load_diffusion_model operations
        for item in data['load_data'].get('load_diffusion_model', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                func_name = item.get('func_name', 'load_diffusion_model')
                operations.append({
                    'type': 'load_diffusion_model',
                    'name': f'Load Diffusion Model: {func_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time'],
                    'func_name': func_name
                })

    for item in data['sampling_data']:
        # Convert perf_counter times to relative seconds from workflow start
        start_time = item['cfg_guider_start_time'] - workflow_start
        end_time = item['cfg_guider_end_time'] - workflow_start
        operations.append({
            'type': 'sampling',
            'name': f'Sampling: {item["model"]} ({item["steps"]} steps)',
            'start': start_time,
            'end': end_time,
            'duration': item['cfg_guider_elapsed_time']
        })

        # Add sampler_sample if it exists
        if 'sampler_sample_start_time' in item and 'sampler_sample_end_time' in item:
            start_time = item['sampler_sample_start_time'] - workflow_start
            end_time = item['sampler_sample_end_time'] - workflow_start
            avg_iter_time = item.get('average_iteration_time', 0)
            iter_per_sec = 1.0 / avg_iter_time if avg_iter_time > 0 else 0
            operations.append({
                'type': 'sampler_sample',
                'name': f'Sampler Sample: {item["model"]} ({item["steps"]} steps)',
                'start': start_time,
                'end': end_time,
                'duration': item['sampler_sample_elapsed_time'],
                'iter_per_sec': iter_per_sec,
                'sec_per_iter': avg_iter_time
            })

    for item in data['vae_data']['encode']:
        if item['valid_timing']:
            # Convert perf_counter times to relative seconds from workflow start
            start_time = item['start_time'] - workflow_start
            end_time = start_time + item['elapsed_time']
            operations.append({
                'type': 'vae_encode',
                'name': 'VAE Encode',
                'start': start_time,
                'end': end_time,
                'duration': item['elapsed_time']
            })

    for item in data['vae_data']['decode']:
        if item['valid_timing']:
            # Convert perf_counter times to relative seconds from workflow start
            start_time = item['start_time'] - workflow_start
            end_time = start_time + item['elapsed_time']
            operations.append({
                'type': 'vae_decode',
                'name': 'VAE Decode',
                'start': start_time,
                'end': end_time,
                'duration': item['elapsed_time']
            })

    # Add clip tokenize operations if they exist
    if 'clip_data' in data:
        for item in data['clip_data'].get('tokenize', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                model_name = item.get('model', 'CLIP')
                operations.append({
                    'type': 'clip_tokenize',
                    'name': f'CLIP Tokenize: {model_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time']
                })

        # Add clip encode operations
        for item in data['clip_data'].get('encode', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                model_name = item.get('model', 'CLIP')
                func_name = item.get('func_name', '')
                operations.append({
                    'type': 'clip_encode',
                    'name': f'CLIP Encode: {model_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time'],
                    'func_name': func_name
                })

    operations.sort(key=lambda x: x['start'])

    # Determine nesting levels for each operation
    def is_contained(op1, op2):
        """Check if op1 is contained within op2"""
        return op1['start'] >= op2['start'] and op1['end'] <= op2['end'] and op1 != op2

    # Calculate nesting level for each operation
    for i, op in enumerate(operations):
        containing_ops = []
        for j, other_op in enumerate(operations):
            if is_contained(op, other_op):
                containing_ops.append(j)
        op['nesting_level'] = len(containing_ops)
        op['index'] = i

    # Base sizes - make bars much larger to use the space better
    base_width = 80
    width_reduction_per_level = 15

    y_position = 0
    for op in operations:
        # Custom hover template for different operation types
        if op['type'] == 'sampler_sample':
            hover_template = (f'<b>{op["name"]}</b><br>' +
                            f'<b>Start</b>: {op["start"]:.3f}s<br>' +
                            f'<b>End</b>: {op["end"]:.3f}s<br>' +
                            f'<b>Duration</b>: {op["duration"]:.3f}s<br>' +
                            f'<b>Iterations/sec</b>: {op["iter_per_sec"]:.2f}<br>' +
                            f'<b>Seconds/iter</b>: {op["sec_per_iter"]:.3f}s<extra></extra>')
        elif (op['type'] in ['clip_encode', 'load_state_dict', 'load_diffusion_model']) and 'func_name' in op:
            hover_template = (f'<b>{op["name"]}</b><br>' +
                            f'<b>Function</b>: {op["func_name"]}<br>' +
                            f'<b>Start</b>: {op["start"]:.3f}s<br>' +
                            f'<b>End</b>: {op["end"]:.3f}s<br>' +
                            f'<b>Duration</b>: {op["duration"]:.3f}s<extra></extra>')
        else:
            hover_template = (f'<b>{op["name"]}</b><br>' +
                            f'<b>Start</b>: {op["start"]:.3f}s<br>' +
                            f'<b>End</b>: {op["end"]:.3f}s<br>' +
                            f'<b>Duration</b>: {op["duration"]:.3f}s<extra></extra>')

        # Calculate line width based on nesting level
        line_width = max(base_width - (op['nesting_level'] * width_reduction_per_level), 20)

        fig.add_trace(
            go.Scatter(
                x=[op['start'], op['end']],
                y=[y_position, y_position],
                mode='lines',
                line=dict(color=colors.get(op['type'], 'gray'), width=line_width),
                name=op['name'],
                hovertemplate=hover_template,
                showlegend=False
            ),
            row=operations_row, col=1
        )

    legend_items = set()
    for op in operations:
        if op['type'] not in legend_items:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color=colors.get(op['type'], 'gray'), width=10),
                    name=op['type'].replace('_', ' ').title(),
                    showlegend=True
                ),
                row=operations_row, col=1
            )
            legend_items.add(op['type'])

    fig.update_xaxes(title_text="Time (seconds from start)", row=operations_row, col=1)
    if has_nvidia_data:
        fig.update_yaxes(title_text="VRAM (MB)", row=1, col=1)
        fig.update_yaxes(title_text="GPU %", row=2, col=1)
        fig.update_yaxes(title_text="Power (W)", row=3, col=1)
    fig.update_yaxes(showticklabels=False, row=operations_row, col=1)

    fig.update_layout(
        title=f"ComfyUI Benchmark - {data['workflow_name']}",
        height=800,
        hovermode='x',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(tickformat='.1f', ticksuffix='s')
    )

    # Add tickformat for all x-axes if nvidia data is present
    if has_nvidia_data:
        fig.update_layout(
            xaxis2=dict(tickformat='.1f', ticksuffix='s'),
            xaxis3=dict(tickformat='.1f', ticksuffix='s'),
            xaxis4=dict(tickformat='.1f', ticksuffix='s')
        )

    return fig


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "benchmark_20250903_222144.json"

    fig = create_benchmark_visualization(json_file)
    fig.show()
    # fig.write_html(f"{json_file.split('.')[0]}_visualization.html")
