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

    nvidia_smi_data = []
    for line in data['nvidia_smi_data']:
        parsed = parse_nvidia_smi_line(line, workflow_start_datetime, workflow_start)
        if parsed:
            nvidia_smi_data.append(parsed)

    # Use relative times starting from 0
    relative_times = [d['relative_time'] for d in nvidia_smi_data]
    memory_used = [d['memory_used'] for d in nvidia_smi_data]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'VRAM Usage - {device_name}', 'Workflow Operations Timeline'),
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Bar(
            x=relative_times,
            y=memory_used,
            name='VRAM Used (MB)',
            marker_color='lightblue',
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

    operations = []
    colors = {
        'load_torch_file': 'purple',
        'model_load': 'orange', 
        'sampling': 'green',
        'vae_encode': 'blue',
        'vae_decode': 'red'
    }

    for item in data['load_torch_file_data']:
        if item['valid_timing']:
            # Convert perf_counter times to relative seconds from workflow start
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

    for item in data['model_load_data']:
        if item['valid_timing']:
            # Convert perf_counter times to relative seconds from workflow start
            start_time = item['start_time'] - workflow_start
            end_time = start_time + item['elapsed_time']
            operations.append({
                'type': 'model_load',
                'name': f'Model Load: {item["model"]}',
                'start': start_time,
                'end': end_time,
                'duration': item['elapsed_time']
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

    operations.sort(key=lambda x: x['start'])

    y_position = 0
    for op in operations:
        fig.add_trace(
            go.Scatter(
                x=[op['start'], op['end']],
                y=[y_position, y_position],
                mode='lines',
                line=dict(color=colors.get(op['type'], 'gray'), width=20),
                name=op['name'],
                hovertemplate=f'<b>{op["name"]}</b><br>' +
                              f'<b>Start</b>: {op["start"]:.3f}s<br>' +
                              f'<b>End</b>: {op["end"]:.3f}s<br>' +
                              f'<b>Duration</b>: {op["duration"]:.3f}s<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
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
                row=2, col=1
            )
            legend_items.add(op['type'])

    fig.update_xaxes(title_text="Time (seconds from start)", row=2, col=1)
    fig.update_yaxes(title_text="VRAM (MB)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

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
        xaxis=dict(tickformat='.1f', ticksuffix='s'),
        xaxis2=dict(tickformat='.1f', ticksuffix='s')
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
