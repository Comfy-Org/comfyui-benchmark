# NOTE: this file was 99% written by Claude Code
import json
import sys
import argparse
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_nvidia_smi_line(line, workflow_start_datetime, workflow_start_time):
    parts = line.strip().split(', ')
    # Total amount may change, but have a sanity check
    if len(parts) < 2:
        return None

    timestamp_dt = datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f')
    # Convert to relative seconds from workflow start
    time_delta = timestamp_dt - workflow_start_datetime
    relative_time = time_delta.total_seconds()
    # Power limit might be [N/A], so need to account for that
    power_limit = float(parts[7]) if parts[7] != '[N/A]' else None

    return {
        'timestamp': timestamp_dt,
        'relative_time': relative_time,
        'memory_used': int(parts[1]),
        'memory_total': int(parts[2]),
        'gpu_utilization': int(parts[3]),
        'memory_utilization': int(parts[4]),
        'power_draw': float(parts[5]),
        'power_instant': float(parts[6]),
        'power_limit': power_limit,
    }


def parse_psutil_line(line, workflow_start_time):
    """Parse psutil data line: 'relative_seconds,used_memory(bytes),total_memory(bytes)'"""
    parts = line.strip().split(',')
    if len(parts) != 3:
        return None
    
    try:
        relative_time = float(parts[0]) - workflow_start_time
        used_memory_mb = int(parts[1]) / (1024 * 1024)  # Convert bytes to MB
        total_memory_mb = int(parts[2]) / (1024 * 1024)  # Convert bytes to MB
        
        return {
            'relative_time': relative_time,
            'used_memory': used_memory_mb,
            'total_memory': total_memory_mb
        }
    except (ValueError, IndexError):
        return None


def extract_operations_from_data(data, workflow_start):
    """Extract operations data from benchmark data for timeline visualization."""
    operations = []
    
    # Handle both old and new data format
    if 'load_data' in data:
        # New format - load_data dictionary
        for item in data['load_data'].get('load_torch_file', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                # Get full filename including extension
                model_name = item['ckpt'].split('\\')[-1]
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

        # New model_unload operations
        for item in data['load_data'].get('model_unload', []):
            if item['valid_timing']:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                model_name = item.get('model', 'Unknown')
                operations.append({
                    'type': 'model_unload',
                    'name': f'Model Unload: {model_name}',
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

        # New patch_model operations (only if duration > 0.01 seconds)
        for item in data['load_data'].get('patch_model', []):
            if item['valid_timing'] and item['elapsed_time'] > 0.01:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                model_name = item.get('model', 'Unknown')
                operations.append({
                    'type': 'patch_model',
                    'name': f'Patch Model: {model_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time']
                })

        # New unpatch_model operations (only if duration > 0.01 seconds)
        for item in data['load_data'].get('unpatch_model', []):
            if item['valid_timing'] and item['elapsed_time'] > 0.01:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                model_name = item.get('model', 'Unknown')
                operations.append({
                    'type': 'unpatch_model',
                    'name': f'Unpatch Model: {model_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time']
                })

    for item in data.get('sampling_data', []):
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

    if 'vae_data' in data:
        for item in data['vae_data'].get('encode', []):
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

        for item in data['vae_data'].get('decode', []):
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

    # Add cache clean operations if they exist (only if duration > 0.01 seconds)
    if 'caches_data' in data:
        for item in data['caches_data'].get('clean_unused', []):
            if item['elapsed_time'] > 0.01:
                start_time = item['start_time'] - workflow_start
                end_time = start_time + item['elapsed_time']
                cache_name = item.get('cache_name', 'Unknown')
                operations.append({
                    'type': 'cache_clean',
                    'name': f'Cache Clean: {cache_name}',
                    'start': start_time,
                    'end': end_time,
                    'duration': item['elapsed_time']
                })

    operations.sort(key=lambda x: x['start'])
    return operations


def create_benchmark_visualization(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    device_info = data['device_info']
    device_name_full = device_info['name']
    # Extract just GPU name for legacy use
    device_name = ' '.join(device_name_full.split(' ')[1:5]) if len(device_name_full.split(' ')) > 4 else device_name_full
    total_vram = device_info['total_vram']

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

    # Parse psutil data if it exists
    psutil_data = []
    if 'psutil_data' in data and data['psutil_data']:
        for line in data['psutil_data']:
            parsed = parse_psutil_line(line, workflow_start)
            if parsed:
                psutil_data.append(parsed)

    # Determine which graphs to show based on available data
    has_nvidia_data = len(nvidia_smi_data) > 0
    has_psutil_data = len(psutil_data) > 0

    # Create device info table data
    device_table_headers = ['Property', 'Value']
    device_table_cells = []
    device_table_cells.append(['GPU', device_name_full])
    device_table_cells.append(['Total VRAM', f'{total_vram:.1f} MB'])
    device_table_cells.append(['Total RAM', f'{device_info.get("total_ram", "N/A"):.1f} MB' if isinstance(device_info.get("total_ram"), (int, float)) else 'N/A'])
    device_table_cells.append(['VRAM State', device_info.get('vram_state', 'N/A')])
    if 'pytorch_version' in device_info:
        device_table_cells.append(['PyTorch Version', device_info['pytorch_version']])
    if 'operating_system' in device_info:
        device_table_cells.append(['Operating System', device_info['operating_system']])
    
    # Add startup_args if available
    if 'startup_args' in data:
        startup_args = data['startup_args']
        for key, value in startup_args.items():
            device_table_cells.append([key, str(value)])

    # Parse initial VRAM usage if available
    initial_vram = None
    if 'nvidia_smi_data_info' in data:
        nvidia_smi_info = data['nvidia_smi_data_info']
        if 'initial_nvidia_smi_query' in nvidia_smi_info and 'nvidia_smi_query_params' in nvidia_smi_info:
            initial_query = nvidia_smi_info['initial_nvidia_smi_query']
            query_params = nvidia_smi_info['nvidia_smi_query_params']
            
            if initial_query and initial_query.strip() and query_params:
                # Split the parameters to find the index of memory.used
                params_list = [p.strip() for p in query_params.split(',')]
                try:
                    memory_used_index = params_list.index('memory.used')
                    initial_parts = initial_query.strip().split(', ')
                    if len(initial_parts) > memory_used_index:
                        initial_vram = int(initial_parts[memory_used_index])
                except (ValueError, IndexError):
                    pass
    
    # Parse initial RAM usage if available
    initial_ram = None
    if 'psutil_data_info' in data:
        psutil_info = data['psutil_data_info']
        if 'initial_psutil_query' in psutil_info:
            initial_query = psutil_info['initial_psutil_query']
            if initial_query and initial_query.strip():
                # Format: 'relative_seconds,used_memory(bytes),total_memory(bytes)'
                parts = initial_query.strip().split(',')
                if len(parts) == 3:
                    try:
                        initial_ram = int(parts[1]) / (1024 * 1024)  # Convert bytes to MB
                    except (ValueError, IndexError):
                        pass
    
    # Prepare data for graphs
    if has_nvidia_data:
        # Use relative times starting from 0
        relative_times = [d['relative_time'] for d in nvidia_smi_data]
        memory_used = [d['memory_used'] for d in nvidia_smi_data]
        gpu_utilization = [d['gpu_utilization'] for d in nvidia_smi_data]
        power_draw = [d['power_draw'] for d in nvidia_smi_data]  # Use average power for main display
        power_instant = [d['power_instant'] for d in nvidia_smi_data]  # Keep instant power for hover
        power_limit = nvidia_smi_data[0]['power_limit'] if nvidia_smi_data else None

    if has_psutil_data:
        psutil_times = [d['relative_time'] for d in psutil_data]
        ram_used = [d['used_memory'] for d in psutil_data]
        ram_total = psutil_data[0]['total_memory'] if psutil_data else 0

    # Create subplot based on available data
    if has_nvidia_data and has_psutil_data:
        # Both NVIDIA and psutil data available
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=('Device Information', 'VRAM Usage', 'RAM Usage', 'GPU Utilization', 'Power Usage', 'Workflow Operations Timeline'),
            row_heights=[0.12, 0.25, 0.14, 0.14, 0.14, 0.21],
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}]]
        )
        operations_row = 6
        vram_row = 2
        ram_row = 3
        gpu_row = 4
        power_row = 5
    elif has_nvidia_data:
        # Only NVIDIA data available
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=('Device Information', 'VRAM Usage', 'GPU Utilization', 'Power Usage', 'Workflow Operations Timeline'),
            row_heights=[0.15, 0.3, 0.15, 0.15, 0.25],
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}]]
        )
        operations_row = 5
        vram_row = 2
        gpu_row = 3
        power_row = 4
    elif has_psutil_data:
        # Only psutil data available
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=('Device Information', 'RAM Usage', 'Workflow Operations Timeline'),
            row_heights=[0.25, 0.35, 0.40],
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}]]
        )
        operations_row = 3
        ram_row = 2
    else:
        # No monitoring data available
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=('Device Information', 'Workflow Operations Timeline'),
            row_heights=[0.3, 0.7],
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}]]
        )
        operations_row = 2

    # Add device info table
    fig.add_trace(
        go.Table(
            header=dict(
                values=device_table_headers,
                fill_color='lightgray',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[[row[0] for row in device_table_cells], [row[1] for row in device_table_cells]],
                fill_color='white',
                align='left',
                font=dict(size=11)
            )
        ),
        row=1, col=1
    )

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
            row=vram_row, col=1
        )

        # Add horizontal line - use row parameter instead of xref/yref
        fig.add_shape(type="line",
                      x0=min(relative_times), x1=max(relative_times),
                      y0=total_vram, y1=total_vram,
                      line=dict(color="red", dash="dash"),
                      row=vram_row, col=1)
        fig.add_annotation(text=f"Total VRAM: {total_vram} MB",
                          x=max(relative_times), y=total_vram,
                          xanchor="right", yanchor="bottom",
                          showarrow=False,
                          row=vram_row, col=1)
        
        # Add initial VRAM usage line if available
        if initial_vram is not None:
            fig.add_shape(type="line",
                          x0=min(relative_times), x1=max(relative_times),
                          y0=initial_vram, y1=initial_vram,
                          line=dict(color="blue", dash="dot"),
                          row=vram_row, col=1)
            fig.add_annotation(text=f"Initial VRAM: {initial_vram} MB",
                              x=max(relative_times) * 0.95, y=initial_vram,
                              xanchor="right", yanchor="top" if initial_vram < total_vram/2 else "bottom",
                              showarrow=False,
                              row=vram_row, col=1)

    # Add RAM Usage if psutil data is available
    if has_psutil_data:
        fig.add_trace(
            go.Scatter(
                x=psutil_times,
                y=ram_used,
                name='RAM Used (MB)',
                mode='lines',
                line=dict(color='darkgreen', width=2),
                fill='tozeroy',
                hovertemplate='<b>Time</b>: %{x:.2f}s<br>' +
                              '<b>RAM</b>: %{y:.0f} MB<br>' +
                              '<b>Percentage</b>: %{customdata:.1f}%<extra></extra>',
                customdata=[(m/ram_total)*100 for m in ram_used]
            ),
            row=ram_row, col=1
        )

        # Add horizontal line for total RAM
        fig.add_shape(type="line",
                      x0=min(psutil_times), x1=max(psutil_times),
                      y0=ram_total, y1=ram_total,
                      line=dict(color="red", dash="dash"),
                      row=ram_row, col=1)
        fig.add_annotation(text=f"Total RAM: {ram_total:.0f} MB",
                          x=max(psutil_times), y=ram_total,
                          xanchor="right", yanchor="bottom",
                          showarrow=False,
                          row=ram_row, col=1)
        
        # Add initial RAM usage line if available
        if initial_ram is not None:
            fig.add_shape(type="line",
                          x0=min(psutil_times), x1=max(psutil_times),
                          y0=initial_ram, y1=initial_ram,
                          line=dict(color="darkgreen", dash="dot"),
                          row=ram_row, col=1)
            fig.add_annotation(text=f"Initial RAM: {initial_ram:.0f} MB",
                              x=max(psutil_times) * 0.95, y=initial_ram,
                              xanchor="right", yanchor="top" if initial_ram < ram_total/2 else "bottom",
                              showarrow=False,
                              row=ram_row, col=1)

    if has_nvidia_data:
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
            row=gpu_row, col=1
        )

        fig.add_shape(type="line",
                      x0=min(relative_times), x1=max(relative_times),
                      y0=100, y1=100,
                      line=dict(color="gray", dash="dash"),
                      row=gpu_row, col=1)
        fig.add_annotation(text="100%",
                          x=max(relative_times), y=100,
                          xanchor="right", yanchor="bottom",
                          showarrow=False,
                          row=gpu_row, col=1)

        # Add Power Usage
        power_percentages = [(p/power_limit)*100 if power_limit else 0 for p in power_draw]
        # Create custom data with both power percentage and instant power
        customdata_power = [[pct, inst] for pct, inst in zip(power_percentages, power_instant)]
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
                              '<b>Instant Power</b>: %{customdata[1]:.1f}W<br>' +
                              '<b>Power Limit %</b>: %{customdata[0]:.1f}%<extra></extra>',
                customdata=customdata_power
            ),
            row=power_row, col=1
        )

        if power_limit:
            fig.add_shape(type="line",
                          x0=min(relative_times), x1=max(relative_times),
                          y0=power_limit, y1=power_limit,
                          line=dict(color="red", dash="dash"),
                          row=power_row, col=1)
            fig.add_annotation(text=f"Power Limit: {power_limit:.0f}W",
                              x=max(relative_times), y=power_limit,
                              xanchor="right", yanchor="bottom",
                              showarrow=False,
                              row=power_row, col=1)

    # Extract operations using the reusable function
    operations = extract_operations_from_data(data, workflow_start)
    
    colors = {
        'load_torch_file': 'purple',
        'model_load': 'orange',
        'model_unload': 'black',
        'load_state_dict': 'brown',
        'load_diffusion_model': 'indigo',
        'sampling': 'green',
        'sampler_sample': 'cyan',
        'vae_encode': 'blue',
        'vae_decode': 'red',
        'clip_tokenize': 'magenta',
        'clip_encode': 'pink',
        'cache_clean': 'gray',
        'patch_model': 'slategray',
        'unpatch_model': 'slategray'
    }

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

    # Use y_position = 0 for all operations to keep them centered
    y_position = 0
    for op in operations:
        # Custom hover template for different operation types
        if op['type'] == 'sampler_sample':
            hover_template = (f'<b>{op["name"]}</b><br>' +
                            f'<b>Start</b>: {op["start"]:.3f}s<br>' +
                            f'<b>End</b>: {op["end"]:.3f}s<br>' +
                            f'<b>Duration</b>: {op["duration"]:.3f}s<br>' +
                            f'<b>Iterations/sec</b>: {op["iter_per_sec"]:.2f} it/s<br>' +
                            f'<b>Seconds/iter</b>: {op["sec_per_iter"]:.3f} s/it<extra></extra>')
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
    
    # Add vertical line at workflow end time with total duration
    workflow_duration = workflow_end - workflow_start
    fig.add_shape(
        type="line",
        x0=workflow_duration, x1=workflow_duration,
        y0=-0.5, y1=0.5,
        line=dict(color="black", dash="dash", width=2),
        row=operations_row, col=1
    )
    
    # Add annotation for total time
    fig.add_annotation(
        text=f"Total: {workflow_duration:.2f}s",
        x=workflow_duration,
        y=0.4,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font=dict(color="black", size=10),
        row=operations_row, col=1
    )

    fig.update_xaxes(title_text="Time (seconds from start)", row=operations_row, col=1)
    if has_nvidia_data:
        fig.update_yaxes(title_text="VRAM (MB)", row=vram_row, col=1)
        fig.update_yaxes(title_text="GPU %", row=gpu_row, col=1, range=[0, 105])  # Set fixed range for GPU utilization
        fig.update_yaxes(title_text="Power (W)", row=power_row, col=1)
    if has_psutil_data:
        fig.update_yaxes(title_text="RAM (MB)", row=ram_row, col=1)
    # Center the workflow operations timeline
    fig.update_yaxes(showticklabels=False, row=operations_row, col=1, range=[-0.5, 0.5])

    fig.update_layout(
        title=f"ComfyUI Benchmark - {data['workflow_name']}",
        height=1080,
        hovermode='x',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=120),  # Add bottom margin to prevent clipping
        xaxis=dict(tickformat='.1f', ticksuffix='s')
    )

    # Add tickformat and link all x-axes based on available data
    if has_nvidia_data and has_psutil_data:
        # All 6 rows with data
        fig.update_layout(
            xaxis2=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis3=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis4=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis5=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis6=dict(tickformat='.1f', ticksuffix='s', matches='x')
        )
    elif has_nvidia_data:
        # 5 rows with nvidia data only
        fig.update_layout(
            xaxis2=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis3=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis4=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis5=dict(tickformat='.1f', ticksuffix='s', matches='x')
        )
    elif has_psutil_data:
        # 3 rows with psutil data only
        fig.update_layout(
            xaxis2=dict(tickformat='.1f', ticksuffix='s', matches='x'),
            xaxis3=dict(tickformat='.1f', ticksuffix='s', matches='x')
        )

    return fig


def create_benchmark_comparison(json_files):
    """Create a comparison visualization of multiple benchmark files showing only operations timelines."""
    import os
    
    # Color palette for different benchmarks
    benchmark_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'olive']
    
    # Load all benchmark data
    benchmarks = []
    max_time = 0
    
    for idx, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        workflow_start = data['benchmark_data']['workflow_start_time']
        workflow_end = data['benchmark_data']['workflow_end_time']
        workflow_duration = workflow_end - workflow_start
        
        # Extract operations
        operations = extract_operations_from_data(data, workflow_start)
        
        # Get benchmark name from filename
        benchmark_name = os.path.splitext(os.path.basename(json_file))[0]
        workflow_name = data.get('workflow_name', 'Unknown')
        
        # If workflow_name is the same as benchmark_name, just use one
        # Otherwise show both
        if benchmark_name == workflow_name:
            display_name = benchmark_name
        else:
            display_name = f"{benchmark_name} - {workflow_name}"
        
        benchmarks.append({
            'name': benchmark_name,
            'display_name': display_name,
            'operations': operations,
            'duration': workflow_duration,
            'workflow_name': workflow_name,
            'color': benchmark_colors[idx % len(benchmark_colors)]
        })
        
        max_time = max(max_time, workflow_duration)
    
    # Create subplot with one row per benchmark
    num_benchmarks = len(benchmarks)
    subplot_titles = [b['display_name'] for b in benchmarks]
    
    # Calculate vertical spacing as a fraction to maintain constant pixel spacing
    # We want about 40 pixels between subplots to avoid title overlap
    fixed_height_per_benchmark = 180  # Must match the value used later
    total_height = 100 + (fixed_height_per_benchmark * num_benchmarks)
    desired_pixel_spacing = 40
    # vertical_spacing is a fraction of the subplot area (not total height)
    # subplot area = total_height - margins - title space
    subplot_area = total_height - 100  # Approximate usable area
    vertical_spacing_fraction = desired_pixel_spacing / subplot_area if num_benchmarks > 1 else 0
    
    fig = make_subplots(
        rows=num_benchmarks, cols=1,
        shared_xaxes=True,
        vertical_spacing=min(vertical_spacing_fraction, 0.1),  # Cap at 0.1 to avoid issues
        subplot_titles=subplot_titles
    )
    
    # Operation type colors (shared across all benchmarks)
    colors = {
        'load_torch_file': 'purple',
        'model_load': 'orange',
        'model_unload': 'black',
        'load_state_dict': 'brown',
        'load_diffusion_model': 'indigo',
        'sampling': 'green',
        'sampler_sample': 'cyan',
        'vae_encode': 'blue',
        'vae_decode': 'red',
        'clip_tokenize': 'magenta',
        'clip_encode': 'pink',
        'cache_clean': 'gray',
        'patch_model': 'slategray',
        'unpatch_model': 'slategray'
    }
    
    # Track legend items to avoid duplicates
    legend_items = set()
    
    # Add operations for each benchmark
    for row_idx, benchmark in enumerate(benchmarks, 1):
        operations = benchmark['operations']
        
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
        
        # Base sizes - keep consistent regardless of number of benchmarks
        # Use same sizes as single visualization to maintain consistency
        base_width = 80  # Same as single visualization
        width_reduction_per_level = 15  # Same as single visualization
        
        y_position = 0
        for op in operations:
            # Custom hover template for different operation types
            if op['type'] == 'sampler_sample':
                hover_template = (f'<b>{op["name"]}</b><br>' +
                                f'<b>Start</b>: {op["start"]:.3f}s<br>' +
                                f'<b>End</b>: {op["end"]:.3f}s<br>' +
                                f'<b>Duration</b>: {op["duration"]:.3f}s<br>' +
                                f'<b>Iterations/sec</b>: {op["iter_per_sec"]:.2f} it/s<br>' +
                                f'<b>Seconds/iter</b>: {op["sec_per_iter"]:.3f} s/it<extra></extra>')
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
            line_width = max(base_width - (op['nesting_level'] * width_reduction_per_level), 10)
            
            # Only show in legend once
            show_legend = op['type'] not in legend_items
            if show_legend:
                legend_items.add(op['type'])
            
            fig.add_trace(
                go.Scatter(
                    x=[op['start'], op['end']],
                    y=[y_position, y_position],
                    mode='lines',
                    line=dict(color=colors.get(op['type'], 'gray'), width=line_width),
                    name=op['type'].replace('_', ' ').title(),
                    legendgroup=op['type'],
                    showlegend=show_legend,
                    hovertemplate=hover_template
                ),
                row=row_idx, col=1
            )
        
        # Add vertical line at workflow end time
        fig.add_shape(
            type="line",
            x0=benchmark['duration'], x1=benchmark['duration'],
            y0=-0.6, y1=0.6,  # Match the new y_range
            line=dict(color="black", dash="dash", width=2),
            row=row_idx, col=1
        )
        
        # Add annotation for total time (on the left of the line)
        fig.add_annotation(
            text=f"Total: {benchmark['duration']:.2f}s",
            x=benchmark['duration'],
            y=0.45,  # Adjusted for new y_range
            xanchor="right",  # Changed from "left" to "right"
            yanchor="bottom",
            showarrow=False,
            font=dict(color="black", size=10),
            row=row_idx, col=1
        )
    
    # Update layout - set x-axis range to longest benchmark
    fig.update_xaxes(title_text="Time (seconds from start)", row=num_benchmarks, col=1, range=[0, max_time])
    
    # Center all workflow timelines and set same x-axis range for all
    # Set y-axis range appropriate for bar thickness
    y_range = [-0.6, 0.6]  # Slightly larger than original to accommodate thick bars without excess space
    for row in range(1, num_benchmarks + 1):
        fig.update_yaxes(showticklabels=False, row=row, col=1, range=y_range)
        if row < num_benchmarks:  # Don't duplicate for the last row (already set above)
            fig.update_xaxes(range=[0, max_time], row=row, col=1)
    
    # Fixed height per benchmark subplot to maintain consistent visual size
    fixed_height_per_benchmark = 180  # Reasonable height for each benchmark subplot
    # Calculate legend position to keep it close to the bottom graph
    # The y position needs to be calculated based on the number of benchmarks
    # to maintain a constant distance from the last subplot
    legend_y_position = -60 / (100 + (fixed_height_per_benchmark * num_benchmarks))
    
    fig.update_layout(
        title="ComfyUI Benchmark Comparison",
        height=100 + (fixed_height_per_benchmark * num_benchmarks),  # Fixed height per benchmark
        hovermode='x',  # Changed from 'x unified' to match single visualization
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y_position,  # Dynamic position based on total height
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=120, l=50, r=50),  # Reduced bottom margin since legend is closer
        xaxis=dict(tickformat='.1f', ticksuffix='s')
    )
    
    # Link all x-axes
    for i in range(2, num_benchmarks + 1):
        fig.update_layout(**{f'xaxis{i}': dict(tickformat='.1f', ticksuffix='s', matches='x')})
    
    return fig


if __name__ == "__main__":
    import os
    
    parser = argparse.ArgumentParser(description='Visualize ComfyUI benchmark results')
    parser.add_argument('benchmark_file', nargs='+', help='Path to benchmark JSON file(s). Multiple files will create a comparison view.')
    parser.add_argument('--save-image', dest='save_image', nargs='?', const='default', 
                        help='Save visualization as an image file (PNG, JPG, SVG, etc.). If no path provided, saves in current directory.')
    parser.add_argument('--save-html', dest='save_html', nargs='?', const='default',
                        help='Save visualization as an HTML file. If no path provided, saves in current directory.')
    parser.add_argument('--no-show', dest='no_show', action='store_true', help='Skip opening the visualization in browser')
    
    args = parser.parse_args()
    
    # Check if we're doing a comparison or single visualization
    if len(args.benchmark_file) == 1:
        # Single benchmark visualization
        fig = create_benchmark_visualization(args.benchmark_file[0])
        base_name = os.path.splitext(os.path.basename(args.benchmark_file[0]))[0]
    else:
        # Multiple benchmark comparison
        print(f"Creating comparison visualization for {len(args.benchmark_file)} benchmarks...")
        fig = create_benchmark_comparison(args.benchmark_file)
        base_name = "benchmark_comparison"
    
    # Save as image if requested
    if args.save_image:
        if args.save_image == 'default':
            image_path = f"{base_name}_visualization.png"
        else:
            image_path = args.save_image
        print(f"Saving image to: {image_path}")
        # For comparison, adjust height dynamically
        if len(args.benchmark_file) > 1:
            fixed_height_per_benchmark = 180  # Same as in create_benchmark_comparison
            height = 100 + (fixed_height_per_benchmark * len(args.benchmark_file))
            fig.write_image(image_path, width=1920, height=height, scale=2)
        else:
            fig.write_image(image_path, width=1920, height=1080, scale=2)
    
    # Save as HTML if requested
    if args.save_html:
        if args.save_html == 'default':
            html_path = f"{base_name}_visualization.html"
        else:
            html_path = args.save_html
        print(f"Saving HTML to: {html_path}")
        fig.write_html(html_path)
    
    # Show the figure unless --no-show is specified
    if not args.no_show:
        fig.show()
    elif not args.save_image and not args.save_html:
        print("Warning: --no-show specified without --save-image or --save-html. No output generated.")
