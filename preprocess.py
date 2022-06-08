from pathlib import Path
import json

profiledir="/home/bzheng/workspace/traces/orig/"
resultdir="/home/bzheng/workspace/traces/"
basepath = Path(profiledir)
files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_file())
for item in files_in_basepath:
    if item.name.find('.trace')!=-1 or item.name.find('.json')!=-1:
        with open(str(basepath)+'/'+item.name) as f:
            suport_format = '''
            {
            "schemaVersion": 1,
            "deviceProperties": [
            ],
            "traceEvents": [
            {
                "ph": "X", "cat": "cpu_op",
                "name": "aten::empty", "pid": 1819166, "tid": 1819166,
                "ts": 1639982860075256, "dur": 5,
                "args": {
                "External id": 2,
                "Trace name": "PyTorch Profiler", "Trace iteration": 0

                }
            }]
            }
            '''
            trace_info = json.load(f)
            trace_event = []
            if isinstance(trace_info, list) and len(trace_info) > 0 and 'ph' in trace_info[0]:
                trace_events= trace_info
            elif isinstance(trace_info, dict) and 'traceEvents' in trace_info:
                trace_events = trace_info['traceEvents'] 
            else:
                logger.error("Unsurported json format. now we only support the format like {}".format(suport_format)) 
                exit()
            start_time=0
            for event in trace_events:
                if event['ph'] != 'X':
                    continue
                if 'cpu_op' in event and event['cat'] != 'cpu_op': #PyTorch Profiler (0) cat:other
                    continue
                if event['name']=='model_inference':
                    start_time=event['ts']
                    continue
                if '#' in event['name']:
                    continue
                event['ts'] -= start_time
                trace_event.append(event)
            with open(str(resultdir)+'/'+str(item.name.split('.')[0])+'.trace', 'w') as fout:
                json.dump(trace_event, fout)
