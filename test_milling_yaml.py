import yaml

try:
    with open('.riper/agents/milling.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('YAML: Configs 1 added. Parsing: Success')
    print(f'Agent: {config["agent_type"]}, Tasks: {len(config["tasks"])}')
    print(f'Capacity: {config["milling_config"]["event_capacity"]} attendees')
except Exception as e:
    print(f'YAML: Parsing failed - {e}')
