import requests
import yaml

from utils import get_env_or_raise, get_timestamp

def submit_job():
    ltp_url = get_env_or_raise('LTP_URL')
    ltp_token = get_env_or_raise('LTP_TOKEN')
    ltp_job_yaml_path = get_env_or_raise('LTP_JOB_YAML_PATH')

    headers = {
        'Authorization': f'Bearer {ltp_token}',
        'Content-Type': 'text/yaml',
    }

    with open(ltp_job_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data['name'] += f'-{get_timestamp()}'

    response = requests.post(ltp_url, headers=headers, data=yaml.dump(data))

    print(f'[LTP Job Submission] Status Code: {response.status_code}')
    print(f'[LTP Job Submission] Response Body: {response.text}')

def main():
    submit_job()

if __name__ == '__main__':
    main()
