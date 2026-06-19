"""
Pull per-job power + CO2 from the Calcul Québec / Alliance job-stats portal
and save <base>/Exp_<N>/train/power_and_co2_metrics.csv (columns: time,power,co2).

  power -- node power draw (W) sampled over the job
  co2   -- total CO2 for the whole job (kg), repeated on every row

AUTH: the portal is behind a login. Log in with a browser, copy your
`sessionid` cookie, and export it before running:
    export COOKIE=<sessionid-value>          # bash
    $env:COOKIE = "<sessionid-value>"         # PowerShell

CLUSTERS: jobs ran on different clusters (e.g. ODT on Rorqual, the rest on
Narval), and each cluster has its own portal host. Run once per host:
    python get_drac_power_usage.py -e 7000 7179 -u hartman -p /storage_1/metrics --host portail.narval.calculquebec.ca
    python get_drac_power_usage.py -e 7000 7179 -u hartman -p /storage_1/metrics --host portail.rorqual.alliancecan.ca
(If a job isn't found on a host it's just skipped, so it's safe to run every
range against every host and let the union fill in.)

Default host stays the historical Beluga portal for backward compatibility.
"""
import json
import os
import argparse
import urllib.request

cookie = f"sessionid={os.environ.get('COOKIE')}"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-CA,en-US;q=0.7,en;q=0.3', 'Connection': 'keep-alive',
    'Cookie': cookie,
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1', 'Cache-Control': 'max-age=0',
}


def _get_json(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)).read())


def save_job_power_usage(job, experiment_num, username, base_path, host):
    """Save the power usage + CO2 of one job to a per-experiment CSV."""
    power_url = f"https://{host}/secure/jobstats/{username}/{job['id_job']}/graph/power.json"
    print(f"Making request to {power_url}")
    power_data = _get_json(power_url)

    co2_url = f"https://{host}/secure/jobstats/{username}/{job['id_job']}/value/cost.json"
    print(f"Making request to {co2_url}")
    co2_data = _get_json(co2_url)

    save_path = f"{base_path}/Exp_{experiment_num}/train/power_and_co2_metrics.csv"
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(power_data['data'][0]['x']) == 0:
        raise Exception(f"No power data found for job {job['id_job']}")
    if 'co2_emissions_kg' not in co2_data:
        raise Exception(f"No CO2 data found for job {job['id_job']}")

    print(f"Writing to {save_path}")
    with open(save_path, 'w') as f:
        f.write('time,power,co2\n')
        for x, y in zip(power_data['data'][0]['x'], power_data['data'][0]['y']):
            f.write(f'{x},{y},{co2_data["co2_emissions_kg"]}\n')

def get_jobs_per_experiments(experiment_list, username, base_path, host, length):
    """Pull power+CO2 for every experiment in [start, end]."""
    url = f"https://{host}/api/jobs/?format=datatables&username={username}&length={length}"
    print(f"Making request to {url}")
    data = _get_json(url)

    saved, skipped = 0, []
    for experiment in range(experiment_list[0], experiment_list[1] + 1):
        # Most recent matching job that actually has power data wins.
        for job in data['data']:
            if job['job_name'] == f"Exp_{experiment}_train":
                try:
                    save_job_power_usage(job, experiment, username, base_path, host)
                    print(f"Saved data for job {job['id_job']}")
                    saved += 1
                    break
                except Exception as e:
                    print(e)
        else:
            skipped.append(experiment)
    print(f"\n[{host}] saved {saved}, no job/power found for {len(skipped)} experiments")
    if skipped:
        print(f"  not found here (may live on another cluster): {skipped[:20]}"
              + (" ..." if len(skipped) > 20 else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the power usage of a job')
    parser.add_argument('-e', type=int, nargs='*', default=[], help='start end (inclusive)')
    parser.add_argument('-u', type=str, default='hartman', help='portal username')
    parser.add_argument('-p', type=str, default='../../../../storage_1/metrics', help='metrics base path')
    parser.add_argument('--host', type=str, default='portail.beluga.calculquebec.ca',
                        help='portal host for the cluster the jobs ran on')
    parser.add_argument('--length', type=int, default=2000, help='max jobs to fetch from the portal')
    args = parser.parse_args()

    if not os.environ.get('COOKIE'):
        raise SystemExit("Set the COOKIE env var to your portal sessionid first "
                         "(see the module docstring).")
    if len(args.e) != 2:
        raise SystemExit("Pass a start and end experiment number, e.g. -e 7000 7179")

    get_jobs_per_experiments(args.e, args.u, args.p, args.host, args.length)
