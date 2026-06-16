import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime, timedelta
import urllib.request
import os
import argparse

cookie = f"sessionid={os.environ.get('COOKIE')}"

def save_job_power_usage(job, experiment_num, username, base_path):
    """
    Save the power usage of a job to a CSV file

    Parameters:
        job (dict): Job information
        experiment_num (int): Experiment number
        username (str): Username
        base_path (str): Base path to save the CSV file
    """

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-CA,en-US;q=0.7,en;q=0.3', 'Connection': 'keep-alive',
                'Cookie': cookie,
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document','Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Site': 'none', 'Sec-Fetch-User': '?1', \
                'Cache-Control': 'max-age=0'}

    power_url = f"https://portail.narval.calculquebec.ca/secure/jobstats/{username}/{job['id_job']}/graph/power.json"
    print(f"Making request to {power_url}")
    power_get = urllib.request.urlopen(urllib.request.Request(power_url, headers=headers))
    power_data = json.loads(power_get.read())

    co2_url = f"https://portail.narval.calculquebec.ca/secure/jobstats/{username}/{job['id_job']}/value/cost.json"
    print(f"Making request to {co2_url}")
    co2_get = urllib.request.urlopen(urllib.request.Request(co2_url, headers=headers))
    co2_data = json.loads(co2_get.read())

    save_path = f"{base_path}/Exp_{experiment_num}/train/power_and_co2_metrics.csv"

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(power_data['data'][0]['x']) < 0:
        raise Exception(f"No power data found for job {job['id']}")
    
    if 'co2_emissions_kg' not in co2_data:
        raise Exception(f"No CO2 data found for job {job['id']}")

    print(f"Writing to {save_path}")
    with open(save_path, 'w') as f:
        f.write('time,power,co2\n')
        for x, y in zip(power_data['data'][0]['x'], power_data['data'][0]['y']):
            f.write(f'{x},{y},{co2_data["co2_emissions_kg"]}\n')

def get_jobs_per_experiments(experiment_list, username, base_path):

    """
    Get the jobs for a list of experiments

    Parameters:
        experiment_list (list): List of experiment numbers
        username (str): Username
        base_path (str): Base path to save the CSV file
    """

    url = f"https://portail.narval.calculquebec.ca/api/jobs/?format=datatables&username={username}&length=2000"

    print(f"Making request to {url}")

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-CA,en-US;q=0.7,en;q=0.3', 'Connection': 'keep-alive',
                'Cookie': cookie,
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document','Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Site': 'none', 'Sec-Fetch-User': '?1', \
                'Cache-Control': 'max-age=0'}

    get = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    data = json.loads(get.read())

    for experiment in range(experiment_list[0], experiment_list[1] + 1):
        # Find most recent id_job for this experiment that has power data
        for job in data['data']:
            if job['job_name'] == f"Exp_{experiment}_train":
                try:
                    save_job_power_usage(job, experiment, username, base_path)
                    print(f"Saved data for job {job['id_job']}")
                    break
                except Exception as e:
                    print(e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get the power usage of a job')
    parser.add_argument('-e', type=int, nargs='*', default=[])
    parser.add_argument('-u', type=str, default='hartman')
    parser.add_argument('-p', type=str, default='../../../../storage_1/metrics')
    args = parser.parse_args()

    get_jobs_per_experiments(args.e, args.u, args.p)
